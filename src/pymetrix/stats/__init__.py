"""Statistics and tests. Each test lives with what it tests and returns a TestResult."""

from .base import Test, TestResult, available_tests, register, run_test
from .compare import compare
from .data_tests import ADF, KPSS, EngleGranger, PhillipsOuliaris
from .panel_tests import CIPS, Breitung, FisherADF, Hadri, ImPesaranShin, LevinLinChu
from .panel_cointegration import Kao, Pedroni, Westerlund
from .equation_tests import (
    ARCH,
    RESET,
    BreuschGodfrey,
    BreuschPagan,
    Chow,
    Endogeneity,
    FirstStage,
    HansenJ,
    JarqueBera,
    LjungBox,
    Sargan,
    Wald,
    White,
    diagnostics,
)
from .forecast import forecast_accuracy
from .density import berkowitz, crps_mixture, crps_normal, crps_sample, log_score_mixture, log_score_normal, log_score_sample, pit_mixture, pit_normal, pit_sample
from .evaluation import (binary_log_score, binary_scores, brier_score, christoffersen, clark_west, encompassing, giacomini_white, kuipers_score,
                         linex_loss, linlin_loss, mincer_zarnowitz, model_confidence_set, pesaran_timmermann, roc_auc, roc_curve, spa, stepm)
from .stability import CUSUM, CUSUMSQ
from .breaks import QuandtAndrews, StructuralBreaks, bai_perron
from .var_tests import Granger, Portmanteau, VARNormality
from .timeseries import OrderSelection, acf, pacf, select_order
from .unit_root_extra import ERSPointOptimal, NgPerron, PerronBreak

__all__ = [
    "ADF", "DFGLS", "KPSS", "EngleGranger", "PhillipsOuliaris", "PhillipsPerron", "VarianceRatio", "ZivotAndrews", "ARCH", "CUSUM", "CUSUMSQ", "QuandtAndrews", "StructuralBreaks", "bai_perron", "Granger", "Portmanteau", "VARNormality", "RESET", "BreuschGodfrey", "BreuschPagan", "Chow", "Endogeneity", "FirstStage",
    "JarqueBera", "LjungBox", "Sargan", "Test", "TestResult", "Wald", "White", "available_tests", "compare",
    "OrderSelection", "acf", "diagnostics", "forecast_accuracy", "pacf", "register", "run_test", "select_order",
    "berkowitz", "crps_mixture", "crps_normal", "crps_sample", "log_score_mixture", "log_score_normal", "log_score_sample", "pit_mixture",
    "pit_normal", "pit_sample", "christoffersen", "clark_west", "encompassing", "giacomini_white", "mincer_zarnowitz", "model_confidence_set", "spa",
    "binary_log_score", "binary_scores", "brier_score", "kuipers_score", "linex_loss", "linlin_loss", "pesaran_timmermann", "roc_auc",
    "roc_curve", "stepm", "NgPerron", "ERSPointOptimal", "PerronBreak",
]


_PLUGIN_TESTS = {"DFGLS", "PhillipsPerron", "VarianceRatio", "ZivotAndrews"}  # in the bundled arch plugin


def __getattr__(name: str):
    if name in _PLUGIN_TESTS:
        from ..ext import arch

        return getattr(arch, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
