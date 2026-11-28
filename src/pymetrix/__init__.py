"""pymetrix: econometrics and economic modelling.

Four verbs cover the package: ``fit`` (specification + data -> fitted object),
``solve`` (model + data -> Solution), ``predict``/``forecast``, and
``linearize``/``steady_state``.

Only working features are exported here. Planned ones (GMM, quantile regression,
principal components and factor models, Monte Carlo, sign restrictions) exist in
their modules as placeholders that raise ``NotImplementedError`` naming the build stage.
"""

from ._version import __version__
from .data import CodeList, Dataset, EntityAxis, Sample, TimeAxis
from .estimation import (
    ARMA,
    GARCH,
    GMM,
    FittedGMM,
    MomentGMM,
    IV,
    ML,
    OLS,
    Bayes,
    Cluster,
    Draws,
    Equation,
    FittedEquation,
    Heckman,
    Logit,
    Multinomial,
    NegativeBinomial,
    OrderedLogit,
    OrderedProbit,
    PointEstimate,
    Poisson,
    Probit,
    Quantile,
    RobustLS,
    Tobit,
    ZeroInflatedPoisson,
    marginal_effects,
    RollingFit,
    quantile_process,
)
from .filters import HP, BaxterKing, ChristianoFitzgerald, Hamilton, MovingAverage
from .seasonal import STL, Classical
from .forecasting import Backtest, ForecastRound, ForecastStore, backtest, combine_forecasts, diebold_mariano
from .report import Report, report_templates
from .demos import demo, demos
from .themes import Theme, get_theme, set_theme, themes
from .threads import get_blas_threads, set_blas_threads
from ._random import get_random_generator, random_generator, set_random_generator
from .disaggregation import FittedDisaggregation, TemporalDisaggregation, aggregation_matrix
from .reconciliation import Reconciliation, reconcile
from .models.dynare import DynareModel, read_mod
from .models.iris import read_iris
from .models.eviews import eviews_series, read_eviews, read_eviews_sspace, read_eviews_xml
from .models.bayes import Identification, Posterior
from .models.sensitivity import StabilityMapping, stability_mapping
from .models.dsge_var import FittedDSGEVAR
from .sweeps import ScenarioSet, SweepResult, compare_solutions, scenarios, sweep
from .models.occbin import OccBinPath, occbin
from .models.occbin_filter import OccBinFilter, occbin_filter
from .models.regimes import Regime, RegimeFilter, parse_regimes, regime_filter
from .models.links import Link
from .models.source import ModelSource
from .models.writers import export_model
from .models.perturbation import SecondOrderSolution, second_order
from .models.particle import ParticleFilter, particle_filter
from .models.policy import OptimalPolicy, OSRResult, optimal_policy, osr
from .models.control import OptimalControl, optimal_control
from .errors import DataError, NotFoundError, PlannedError, PyMetrixError, SpecificationError
from .stats.breaks import StructuralBreaks, bai_perron
from .treatment import DiD, RDD, EventStudy, SyntheticControl, TreatmentEffect, did, event_study, rdd, synthetic_control, treatment_effect
from .estimation import priors
from .midas import MIDAS, FittedMIDAS
from .nowcast import Bridge, DynamicFactor, FittedBridge, FittedDynamicFactor, News
from . import nowcast  # nowcast.combine and the ragged-edge tools, as scripts name them
from .factors import PCA, FactorModel, FittedFactorModel, FittedPCA
from .ardl import ARDL, FittedARDL
from .markov import FittedMarkovSwitching, MarkovSwitching
from .metadata import Adjustment, DatasetMeta, Measure, Param, Prices, Provenance, Role, Status, Variable, Vintage
from .models import Fallback, FittedModel, GaussSeidel, LinearModel, MeasurementMap, Model, ModelCheck, ModelError, ModelSystem, Newton, Parameter, ParameterSet, PerfectForesight, FirstOrder, Scenario, Solution, templates
from .statespace import StateSpaceSystem
from .systems import FIML
from .systems import SUR, FittedSystem, System, SystemOLS, ThreeSLS, TwoSLS
from .bvar import FittedBVAR, Minnesota, NormalWishart
from .var_conditions import Bounds
from .bvar_hierarchical import GLP
from .statespace import FittedStateSpace, StateSpace, UnobservedComponents
from .ets import ExponentialSmoothing, FittedExponentialSmoothing
from .matrix import Matrix
from .arima import SARIMA, STLForecast, Theta, FittedSARIMA, FittedSTLForecast, FittedTheta
from .bma import BMA, FittedBMA
from .nonlinear import FittedThresholdAR, ThresholdAR, terasvirta
from .smooth_transition import FittedSmoothTransitionAR, SmoothTransitionAR
from .proxy import ProxySVAR
from .tvpvar import TVPVAR, FittedTVPVAR
from .bvar_sv import BVARSV, FittedBVARSV
from .favar import FittedFAVAR, favar
from .ms_var import MSVAR, FittedMSVAR
from .quantile_var import FittedQuantileVAR, QuantileVAR
from .spatial import SpatialModel, knn_weights, spatial_diagnostics, spatial_regression
from .spectral import CrossSpectrum, FittedARFIMA, Spectrum, arfima, cross_spectrum, gph, local_whittle, periodogram, spectrum
from .structural import AB, StructuralEstimate
from .identification_extra import HeteroskedasticitySVAR, MaxShare
from .estimation.direct import ols
from .estimation.regression_extras import CochraneOrcutt, MultiplicativeHeteroskedasticity, PraisWinsten, StochasticFrontier, cox_test, j_test
from .nonparametric import (KernelDensity, KernelRegression, PartiallyLinear, SemiparametricGARCH, kde, kernel_regression,
                            nonparametric_autoregression, partially_linear, semiparametric_garch)
from .count_series import SeriesModel, acd, ach, inar, poisson_autoregression
from .garch_in_mean import garch_in_mean
from .nonlinear_extra import FittedThresholdVAR, bilinear, threshold_var
from .varma import FittedVARMA, varma
from .synthdid import SyntheticDiD, synthetic_did
from .cointegrating_regression import CointegratingRegression, cointegrating_regression, gregory_hansen
from .panel_time_series import PanelMeanGroup, PooledMeanGroup, cce, mean_group, pooled_mean_group
from .diffusion import diffusion
from .estimation.gmm import MinimumDistance, anderson_rubin, anderson_rubin_set, c_test, minimum_distance, stock_wright
from .simulation import MonteCarlo, MonteCarloResult
from .simulation_estimators import SimulationEstimate, efficient_method_of_moments, indirect_inference, simulated_moments
from .staggered import BaconDecomposition, StaggeredDiD, bacon_decomposition, staggered_did
from .dml import FittedDoubleML, double_ml
from .shiftshare import FittedShiftShare, shift_share
from .interactive_fe import FittedInteractiveFE, interactive_fe
from .clogit import FittedConditionalLogit, choice_logit, conditional_logit
from .clogit_effects import ConditionalLogitEffects, clogit_effects
from .projections import FittedLocalProjection, LocalProjection
from .smooth_lp import FittedSmoothLocalProjection
from .growth_at_risk import GrowthAtRisk, growth_at_risk
from .var import FEVD, IRF, VAR, Cholesky, FittedVAR, LongRun, Restricted, SignRestrictions
from .vecm import VECM, FittedVECM, JohansenResult, johansen
from .output import Figure, Table, show
from .data.coerce import as_dataset
from .selection import as_selection, fit_each, test_each
from .figures import coefficient_figure, heatmap_figure, house_style, set_house_style, surface_figure
from .saved import SavedResult, load_result, save_result
from .stats import TestResult, binary_scores, compare, forecast_accuracy, model_confidence_set, spa
from .plugins import PluginInfo, plugin_table, registry
from .selection import Results, Selection
from . import toolkit

# bundled reference plugins (they register their methods, variances and tests on import)
from .ext.arch import APARCH, EGARCH, FIGARCH, TARCH
from .ext.linearmodels import IVGMM, LIML, Between, FamaMacBeth, FirstDifference, RandomEffects
from .ext.sklearn import Bagging, ElasticNet, FittedLearner, GradientBoosting, Lasso, NearestNeighbours, NeuralNetwork, RandomForest, Ridge
from .ext.connectors import register_sdmx_provider  # and the fred://, worldbank:// and sdmx:// readers
from .views import _attach_all, figure_of
from .recessions import recession_series, recessions

dataset = as_dataset  # pymetrix.dataset(df): a plain DataFrame as a Dataset

_attach_all()  # every result type with views: to_frame(view) and a notebook display (pymetrix.views)
del _attach_all

__all__ = [
    "FittedDoubleML", "double_ml", "FittedShiftShare", "shift_share", "FittedInteractiveFE", "interactive_fe", "FittedConditionalLogit",
    "choice_logit", "conditional_logit", "ConditionalLogitEffects", "clogit_effects",
    "VECM", "FittedVECM", "JohansenResult", "johansen", "FEVD", "IRF", "Cholesky", "FittedVAR", "LongRun", "Restricted", "SignRestrictions", "HP", "ARDL", "FittedARDL", "Matrix", "SARIMA", "FittedSARIMA", "Theta", "FittedTheta", "STLForecast", "FittedSTLForecast", "MarkovSwitching", "FittedMarkovSwitching", "MIDAS", "FittedMIDAS", "Bridge", "FittedBridge", "DynamicFactor", "FittedDynamicFactor", "News", "nowcast", "PCA", "FittedPCA", "FactorModel", "FittedFactorModel", "read_mod", "DynareModel", "read_iris", "read_eviews", "read_eviews_xml", "read_eviews_sspace", "eviews_series", "Posterior", "Identification", "StabilityMapping", "stability_mapping", "priors", "sweep", "scenarios", "compare_solutions", "SweepResult", "ScenarioSet", "occbin", "OccBinPath", "occbin_filter", "OccBinFilter", "Regime", "RegimeFilter", "parse_regimes", "regime_filter", "Link", "ModelSource", "export_model", "second_order", "SecondOrderSolution", "particle_filter", "ParticleFilter", "optimal_policy", "osr", "OptimalPolicy", "OSRResult", "optimal_control", "OptimalControl", "PyMetrixError", "DataError", "NotFoundError", "SpecificationError", "PlannedError", "Quantile", "quantile_process", "RobustLS", "Report", "report_templates", "demo", "demos", "Theme", "get_theme", "set_theme", "themes", "get_blas_threads", "set_blas_threads", "get_random_generator", "set_random_generator", "random_generator", "TemporalDisaggregation", "FittedDisaggregation", "aggregation_matrix", "reconcile", "Reconciliation", "register_sdmx_provider", "Backtest", "ForecastRound", "ForecastStore", "backtest", "combine_forecasts", "diebold_mariano", "ExponentialSmoothing", "FittedExponentialSmoothing", "BMA", "FittedBMA", "ThresholdAR", "FittedThresholdAR", "terasvirta", "Bagging", "ElasticNet", "FittedLearner", "GradientBoosting", "Lasso", "NearestNeighbours", "NeuralNetwork", "RandomForest", "Ridge", "STL", "Classical", "BaxterKing", "ChristianoFitzgerald", "Hamilton", "MovingAverage",
    "APARCH", "ARMA", "EGARCH", "FIGARCH", "TARCH", "GARCH", "IVGMM", "LIML", "RandomEffects", "Between", "FirstDifference", "FamaMacBeth", "IV", "GMM", "MomentGMM", "FittedGMM", "ML", "OLS", "ols", "bai_perron", "StructuralBreaks", "did", "DiD", "event_study", "EventStudy", "rdd", "RDD", "treatment_effect", "TreatmentEffect", "synthetic_control", "SyntheticControl", "Heckman", "Multinomial", "NegativeBinomial", "OrderedLogit", "OrderedProbit", "Tobit", "ZeroInflatedPoisson", "marginal_effects", "VAR", "Bounds", "Adjustment", "Bayes", "Cluster",
    "as_dataset", "dataset", "as_selection", "fit_each", "test_each", "CodeList", "Dataset", "DatasetMeta", "Sample", "Draws", "EntityAxis", "Equation", "Figure", "figure_of", "coefficient_figure", "heatmap_figure", "surface_figure", "house_style", "set_house_style", "recessions", "recession_series", "FittedEquation", "SavedResult", "load_result", "save_result", "Fallback", "GaussSeidel", "LinearModel", "Logit", "Measure", "Model", "ModelCheck", "ModelError", "ModelSystem", "Parameter", "ParameterSet", "MeasurementMap", "StateSpaceSystem", "FittedModel", "FittedDSGEVAR", "templates",
    "Newton", "Param", "PerfectForesight", "FirstOrder", "PointEstimate", "Poisson", "Prices", "Probit",
    "Provenance", "Role", "RollingFit", "Scenario", "Solution", "StateSpace", "FittedStateSpace", "UnobservedComponents", "LocalProjection", "FittedLocalProjection", "FittedSmoothLocalProjection", "SmoothTransitionAR", "FittedSmoothTransitionAR", "ProxySVAR", "TVPVAR", "FittedTVPVAR", "BVARSV", "FittedBVARSV", "FittedFAVAR", "favar", "MSVAR", "FittedMSVAR", "FittedQuantileVAR", "QuantileVAR", "SpatialModel", "knn_weights", "spatial_diagnostics", "spatial_regression", "CrossSpectrum", "FittedARFIMA", "Spectrum", "arfima", "cross_spectrum", "gph", "local_whittle", "periodogram", "spectrum", "AB", "StructuralEstimate", "MaxShare", "HeteroskedasticitySVAR", "CochraneOrcutt", "PraisWinsten", "MultiplicativeHeteroskedasticity", "StochasticFrontier", "j_test", "cox_test", "KernelDensity", "KernelRegression", "PartiallyLinear", "SemiparametricGARCH", "kde", "kernel_regression", "nonparametric_autoregression", "partially_linear", "semiparametric_garch", "SeriesModel", "acd", "ach", "inar", "poisson_autoregression", "garch_in_mean", "FittedThresholdVAR", "bilinear", "threshold_var", "FittedVARMA", "varma", "SyntheticDiD", "synthetic_did", "CointegratingRegression", "cointegrating_regression", "gregory_hansen", "PanelMeanGroup", "PooledMeanGroup", "cce", "mean_group", "pooled_mean_group", "diffusion", "MonteCarlo", "MonteCarloResult", "SimulationEstimate", "MinimumDistance", "minimum_distance", "c_test", "anderson_rubin", "anderson_rubin_set", "stock_wright", "indirect_inference", "simulated_moments", "efficient_method_of_moments", "staggered_did", "StaggeredDiD", "bacon_decomposition", "BaconDecomposition", "growth_at_risk", "GrowthAtRisk", "Status", "System", "FittedSystem", "SUR", "ThreeSLS", "FIML", "TwoSLS", "SystemOLS", "Minnesota", "NormalWishart", "GLP", "FittedBVAR", "Table",
    "Selection", "PluginInfo", "toolkit", "show", "Results", "TestResult", "TimeAxis", "plugin_table", "registry", "Variable", "Vintage", "compare", "forecast_accuracy", "model_confidence_set", "spa", "binary_scores", "__version__",
]


# the finance plugin's names, when it is installed (pip install pymetrix[finance]): from pymetrix import value_at_risk
_FINANCE = ('AbnormalReturns', 'BlackLittermanPosterior', 'CDSCurve', 'FactorModel', 'FittedCopula', 'FittedDCCCopula', 'FittedDNS', 'FittedHAR', 'FittedMGARCH', 'FittedTVCopula', 'FittedVine', 'FittedVolSurface', 'MertonModel', 'OptionValuation', 'Performance', 'Portfolio', 'RiskContributions', 'RiskForecast', 'StrategyBacktest', 'VolatilitySmile', 'YieldCurve', 'abnormal_returns', 'backtest_strategy', 'black_litterman', 'bond_measures', 'cds_curve', 'compare_copulas', 'copula', 'dcc_copula', 'dynamic_nelson_siegel', 'factor_model', 'har', 'heston_price', 'implied_volatility', 'merton', 'merton_point', 'mgarch', 'optimize_portfolio', 'option_price', 'performance', 'realized_measures', 'returns', 'risk_contributions', 'risk_parity_weights', 'tv_copula', 'value_at_risk', 'vine_copula', 'vol_surface', 'yield_curve')


_SEASONAL = ("TramoSeats", "X13")  # the pymetrix-seasonal plugin's methods: X-13ARIMA-SEATS and TRAMO-SEATS


def __getattr__(name: str):
    if name in _SEASONAL:
        try:
            import pymetrix_seasonal
        except ImportError:
            raise AttributeError(f'pymetrix.{name} is in the seasonal-adjustment plugin: pip install "pymetrix-seasonal[jdemetra]"') from None
        return getattr(pymetrix_seasonal, name)
    if name in _FINANCE:
        try:
            import pymetrix_finance
        except ImportError:
            raise AttributeError(f"pymetrix.{name} is in the finance plugin: pip install pymetrix[finance]") from None
        return getattr(pymetrix_finance, name)
    raise AttributeError(f"module 'pymetrix' has no attribute {name!r}")
