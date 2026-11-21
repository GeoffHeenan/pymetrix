from .covariance import Cluster
from .equation import Equation, Term
from .families import Gaussian, Logit, Multinomial, NegativeBinomial, OrderedLogit, OrderedProbit, Poisson, Probit, Tobit, ZeroInflatedPoisson
from .fitted import FittedEquation
from .methods import (
    IV,
    OLS,
    Bayes,
    Calibrated,
)
from .gmm import GMM, FittedGMM, MomentGMM  # after OLS and IV: the methods are listed in the order registered
from .gmm import MinimumDistance, anderson_rubin, anderson_rubin_set, c_test, minimum_distance, stock_wright
from .limited import Heckman, marginal_effects
from .dynpanel import ArellanoBond
from .quantile import Quantile, QuantileProcess, quantile_process
from .robust import RobustLS
from .ml import ML
from .regression_extras import CochraneOrcutt, MultiplicativeHeteroskedasticity, PraisWinsten, StochasticFrontier, cox_test, j_test
from .params import Draws, Params, PointEstimate
from .priors import Beta, Gamma, InvGamma, InvGamma1, InvGamma2, Normal, Uniform
from .rolling import RollingFit
from .direct import ols
from .registry import available_methods, method_settings
from .timeseries import ARMA, GARCH

__all__ = [
    "APARCH", "ARMA", "EGARCH", "FIGARCH", "GARCH", "TARCH", "GMM", "MomentGMM", "FittedGMM", "IV", "IVGMM", "LIML", "ML", "OLS", "ArellanoBond", "Bayes", "Beta", "Between", "Calibrated",
    "Cluster", "Draws", "Equation", "FamaMacBeth", "FirstDifference", "FittedEquation", "Gamma", "Gaussian",
    "Heckman", "marginal_effects", "NegativeBinomial", "OrderedLogit", "OrderedProbit", "ZeroInflatedPoisson", "InvGamma", "InvGamma1", "InvGamma2", "Logit", "Multinomial", "Normal", "Params", "PointEstimate", "Poisson", "Probit", "Quantile", "QuantileProcess", "quantile_process", "RobustLS",
    "RandomEffects", "RollingFit", "Term", "Tobit", "Uniform", "CochraneOrcutt", "PraisWinsten", "MultiplicativeHeteroskedasticity",
    "StochasticFrontier", "j_test", "cox_test", "ols",
]

_PLUGIN_NAMES = {
    "linearmodels": {"LIML", "IVGMM", "RandomEffects", "Between", "FirstDifference", "FamaMacBeth"},
    "arch": {"EGARCH", "TARCH", "FIGARCH", "APARCH"},
}


def __getattr__(name: str):
    """Names provided by the bundled reference plugins (``pymetrix.ext``)."""
    import importlib

    for module, names in _PLUGIN_NAMES.items():
        if name in names:
            return getattr(importlib.import_module(f"..ext.{module}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

