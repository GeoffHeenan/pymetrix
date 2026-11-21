"""The arch adapter: volatility models (``pymetrix.ext.arch``), the unit-root and cointegration tests arch provides, and
its bootstrap comparisons of forecasts (SPA, the model confidence set, StepM)."""

from __future__ import annotations

from typing import Any

from .base import Adapter, AdapterSpec, Option

ADAPTER = Adapter("arch")

_INPUTS = {"y": "y of the mean model", "X (pymetrix's design, with its constant)": "x of LS(constant=False); ConstantMean when "
           "the only regressor is the constant; ZeroMean when there is none"}
_COMMON = (Option("dist", None, "normal → Normal, t → StudentsT, skewt → SkewStudent, ged → GeneralizedError (the mean model's "
                  "distribution)"),)
_METHOD = ("The estimation method's settings map too: `cov` → `fit.cov_type` (classic → classic, white → robust; others refused), "
           "`max_iter` → `fit.options['maxiter']` (at least 100).")
_OUTPUTS = {"params": "params (mu → const)", "param_cov": "params (covariance)", "resid": "resid", "conditional_volatility":
            "extra['volatility']", "loglikelihood": "stats['loglike']", "aic": "stats['aic'] (arch's total form)",
            "bic": "stats['bic']", "nobs": "stats['nobs']", "convergence_flag": "provenance.converged, messages"}
_FIXED = {"fit.disp": ("off", "no optimiser output on the console")}


def _spec(wraps: str, *options: Option) -> AdapterSpec:
    return AdapterSpec(wraps, "arch", _INPUTS, (*options, *_COMMON), _FIXED, _OUTPUTS, notes=_METHOD)


_P, _O, _Q = Option("arch", "p", stage="model"), Option("asym", "o", stage="model"), Option("garch", "q", stage="model")
EGARCH = _spec("arch.univariate.EGARCH", _P, _O, _Q)
TARCH = _spec("arch.univariate.GARCH (power=1)", _P, _O, _Q)
FIGARCH = _spec("arch.univariate.FIGARCH", _P, _Q, Option("power", "power", stage="model"), Option("truncation", "truncation", stage="model"))
APARCH = _spec("arch.univariate.APARCH", _P, _O, _Q, Option("delta", "delta", stage="model", note="empty: estimated"))
GARCH = _spec("arch.univariate.GARCH", _P, _O, _Q)  # pymetrix's GARCH with backend="arch"



def _test(wraps: str, *options: Option, inputs: dict[str, str] | None = None, field: str = "variable") -> AdapterSpec:
    return AdapterSpec(wraps, "arch", inputs or {"the series (missing values dropped)": "y"},
                       (Option(field, None, "the series pymetrix takes from the dataset"), *options),
                       outputs={"stat": "statistic", "pvalue": "p_value", "critical_values": "critical_values",
                                "lags": "details['lags used']", "nobs": "details['observations']"},
                       default_stage="test", stages=("test",), notes="arch computes when the statistic is read; that is inside the run.")


_TREND, _LAGS = Option("trend", "trend"), Option("lags", "lags", "empty → None: arch's default")
_METHOD_LAGS = (Option("method", "method"), Option("max_lags", "max_lags", "empty → None: arch's default"))
PHILLIPS_PERRON = _test("arch.unitroot.PhillipsPerron", _TREND, Option("test_type", "test_type"), _LAGS)
ZIVOT_ANDREWS = _test("arch.unitroot.ZivotAndrews", _TREND, Option("trim", "trim"), _LAGS, *_METHOD_LAGS)
DFGLS = _test("arch.unitroot.DFGLS", _TREND, _LAGS, *_METHOD_LAGS)
VARIANCE_RATIO = _test("arch.unitroot.VarianceRatio", Option("lags", "lags"), _TREND, Option("debiased", "debiased"),
                       Option("robust", "robust"), Option("overlap", "overlap"))
PHILLIPS_OULIARIS = _test("arch.unitroot.cointegration.phillips_ouliaris", _TREND, Option("test_type", "test_type"),
                          Option("kernel", "kernel"), Option("bandwidth", "bandwidth", "empty → None: chosen from the data"),
                          inputs={"the first variable": "y", "the others": "x"}, field="variables")

_BOOT = {"losses": "losses (benchmark first, where there is one)"}
_BOOT_OPTIONS = (Option("reps", "reps", stage="model"), Option("block", "block_size", "default: the larger of the horizon and n^(1/3)", stage="model"),
                 Option("seed", "seed", stage="model"), Option("horizon", None, "sets the default block length"))
SPA = AdapterSpec("arch.bootstrap.SPA", "arch", _BOOT, _BOOT_OPTIONS, outputs={"pvalues": "p_value and details"})
MCS = AdapterSpec("arch.bootstrap.MCS", "arch", _BOOT, (Option("alpha", "size", stage="model"), *_BOOT_OPTIONS),
                  {"model.method": ("R", "the range statistic")}, outputs={"pvalues": "MCS p-value", "included": "in the set"})
STEPM = AdapterSpec("arch.bootstrap.StepM", "arch", _BOOT, (Option("alpha", "size", stage="model"), *_BOOT_OPTIONS),
                    outputs={"superior_models": "superior"})

SPECS = {"EGARCH": EGARCH, "TARCH": TARCH, "FIGARCH": FIGARCH, "APARCH": APARCH, "GARCH": GARCH,
         "PhillipsPerron": PHILLIPS_PERRON, "ZivotAndrews": ZIVOT_ANDREWS, "DFGLS": DFGLS, "VarianceRatio": VARIANCE_RATIO,
         "PhillipsOuliaris": PHILLIPS_OULIARIS, "spa": SPA, "model_confidence_set": MCS, "stepm": STEPM}


def zivot_andrews_all_stats(result: Any) -> Any:
    """The break-date statistics of a Zivot-Andrews test: arch keeps them in a private attribute, so they are read
    here only, with a clear error if a new arch moves them."""
    try:
        return result._all_stats
    except AttributeError:
        from ..errors import PyMetrixError

        raise PyMetrixError("this version of arch no longer provides the Zivot-Andrews break statistics "
                            "(ZivotAndrews._all_stats); report it to pymetrix") from None
