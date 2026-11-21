"""The statsmodels adapter: quantile regression, exponential smoothing, seasonal ARIMA, theta, STL forecasting, Markov
switching, ARDL and VECM; the ADF, KPSS, Engle-Granger, Fisher ADF and Johansen tests and VECM lag selection; seasonal
adjustment (STL, classical and X-13); and dynamic factor nowcasts.

The Kalman filter engine and ``interop.from_statsmodels`` stay outside the adapter by design (``OUTSIDE``).
"""

from __future__ import annotations

from typing import Any

from .._api import renamed_arguments
from .base import Adapter, AdapterSpec, Option

ADAPTER = Adapter("statsmodels")

QUANTREG = AdapterSpec(
    "statsmodels.regression.quantile_regression.QuantReg", "statsmodels", {"y": "endog", "X (with const)": "exog"},
    (Option("q", "q"), Option("vcov", "vcov"), Option("kernel", "kernel"), Option("bandwidth", "bandwidth")),
    {"fit.max_iter": (5000, "statsmodels' default (1000) stops short on some data; raise it with backend_options")},
    {"params": "params", "cov_params()": "params (covariance)", "prsquared": "stats['pseudo_r2']"},
    notes="Residuals and the remaining statistics are computed by pymetrix from the coefficients.")

ETS = AdapterSpec(
    "statsmodels.tsa.exponential_smoothing.ets.ETSModel", "statsmodels", {"the series (unbroken run in the sample)": "endog"},
    (Option("series", None, "the series pymetrix takes from the dataset"), Option("error", "error", stage="model"),
     Option("trend", "trend", "none → None", stage="model"), Option("damped", "damped_trend", stage="model"),
     Option("seasonal", "seasonal", "none → None; seasonal_periods from the frequency", stage="model")),
    {"fit.disp": (False, "no optimiser output on the console")},
    {"params": "params_table", "fittedvalues": "fitted", "states": "states", "llf": "loglike", "aic": "aic", "bic": "bic",
     "mle_retvals['converged']": "provenance.converged"})

MARKOV = AdapterSpec(
    "statsmodels.tsa.regime_switching (MarkovRegression, or MarkovAutoregression when ar > 0)", "statsmodels",
    {"y": "endog", "regressors": "exog", "constant": "trend ('c' or 'n')"},
    (Option("formula", None, "evaluated by pymetrix into y and the regressors"), Option("regimes", "k_regimes", stage="model"),
     Option("ar", "order", "MarkovAutoregression only", stage="model"), Option("switching_variance", "switching_variance", stage="model"),
     Option("switching_exog", "switching_exog", stage="model"), Option("search", "search_reps"),
     Option("seed", "rng", "statsmodels 0.15+; earlier versions draw from NumPy's global generator, seeded and restored")),
    {"fit.disp": (False, "no optimiser output on the console")},
    {"params": "params", "cov_params()": "params (covariance)", "smoothed_marginal_probabilities": "probabilities()",
     "regime_transition": "transition", "mle_retvals['converged']": "provenance.converged"})

ARDL = AdapterSpec(
    "statsmodels.tsa.ardl.ARDL (or ardl_select_order when select is set)", "statsmodels",
    {"y": "endog", "regressors": "exog"},
    (Option("formula", None, "evaluated by pymetrix into y and the regressors"), Option("lags", "lags", stage="model"),
     Option("order", "order", stage="model"), Option("trend", "trend", stage="model"), Option("select", "ic", stage="model"),
     Option("max_lags", "maxlag", stage="model"), Option("max_order", "maxorder", stage="model")),
    outputs={"params": "params", "cov_params()": "params (covariance)", "resid": "resid", "llf": "stats['loglike']",
             "aic": "stats['aic'] (statsmodels' form)", "bic": "stats['bic']"})

VECM = AdapterSpec(
    "statsmodels.tsa.vector_ar.vecm.VECM", "statsmodels", {"the variables": "endog", "exog": "exog", "exog_coint": "exog_coint"},
    (Option("variables", None, "the series pymetrix takes from the dataset"), Option("exog", None, "evaluated by pymetrix"),
     Option("exog_coint", None, "evaluated by pymetrix"), Option("lags", "k_ar_diff", stage="model"),
     Option("rank", "coint_rank", "a number, or chosen by Johansen tests (trace or maxeig)", stage="model"),
     Option("deterministic", "deterministic", stage="model"), Option("seasons", "seasons", stage="model"),
     Option("first_season", "first_season", stage="model"), Option("signif", None, "significance of the Johansen tests that choose the rank")),
    outputs={"alpha": "alpha", "beta": "beta", "gamma": "gamma", "sigma_u": "sigma", "llf": "stats['loglike']", "nobs": "stats['nobs']"})

SARIMAX = AdapterSpec(
    "statsmodels.tsa.statespace.sarimax.SARIMAX", "statsmodels", {"the series (unbroken run in the sample)": "endog", "exog": "exog"},
    (Option("series", None, "the series pymetrix takes from the dataset"), Option("exog", None, "the regressors, from the dataset"),
     Option("order", "order", stage="model"), Option("seasonal_order", "seasonal_order", "with the period: (P, D, Q, s)", stage="model"),
     Option("period", None, "s in seasonal_order; 0 = from the frequency (4 or 12)"),
     Option("trend", "trend", "auto → 'c' without differencing, 'n' with it", stage="model")),
    {"fit.disp": (False, "no optimiser output on the console"),
     "fit.cov_type": ("approx", "standard errors from the numerical Hessian, as R's arima (statsmodels' default is the outer product of gradients)"),
     "fit.method": ("bfgs", ("BFGS (statsmodels' default L-BFGS can stop short of the maximum on flat likelihoods, by an amount "
                             "that depends on the CPU)"))},
    {"params": "params", "cov_params()": "params (covariance)", "fittedvalues": "fitted", "resid": "resid", "llf": "stats['loglike']",
     "get_prediction": "forecast", "mle_retvals['converged']": "provenance.converged"})

THETA = AdapterSpec(
    "statsmodels.tsa.forecasting.theta.ThetaModel", "statsmodels", {"the series (unbroken run in the sample)": "endog"},
    (Option("series", None, "the series pymetrix takes from the dataset"), Option("deseasonalize", "deseasonalize", stage="model"),
     Option("period", "period", "0 → None: from the frequency", stage="model"), Option("method", "method", stage="model")),
    outputs={"params": "params (b0 and alpha)", "forecast(theta=2)": "forecast", "prediction_intervals": "forecast (se from the interval)"})

STL_FORECAST = AdapterSpec(
    "statsmodels.tsa.forecasting.stl.STLForecast", "statsmodels", {"the series (unbroken run in the sample)": "endog"},
    (Option("series", None, "the series pymetrix takes from the dataset"),
     Option("model", "model", "arima → statsmodels.tsa.arima.model.ARIMA, ets → ETSModel", stage="model"),
     Option("order", None, "the ARIMA's order, in model_kwargs"), Option("trend", None, "the ETS trend, in model_kwargs"),
     Option("damped", None, "the ETS damping, in model_kwargs"), Option("period", "period", "0 → from the frequency", stage="model")),
    outputs={"model_result.params": "params", "get_prediction": "forecast", "result.seasonal": "seasonal"})

# ADF's and Engle-Granger's simulated critical values: computed by pymetrix (pymetrix.stats.adf_arrays), not statsmodels
_SIMULATED_OPTIONS = (
    Option("critical_values", None, "mackinnon: statsmodels' (MacKinnon); simulated: pymetrix simulates them, without statsmodels"),
    Option("draws", None, "random walks pymetrix simulates with critical_values='simulated'"),
    Option("seed", None, "the seed of those draws (pymetrix's random generator)"))

ADF = AdapterSpec(
    "statsmodels.tsa.stattools.adfuller", "statsmodels", {"the series (missing values dropped)": "x"},
    (Option("variable", None, "the series pymetrix takes from the dataset"),
     Option("lags", "maxlag", "a number: that many lags (autolag=None); aic / bic / t-stat: autolag, searching up to max_lags"),
     Option("max_lags", "maxlag", "with lags chosen by aic, bic or t-stat"), Option("trend", "regression"),
     *_SIMULATED_OPTIONS),
    outputs={"adf": "statistic", "pvalue": "p_value", "usedlag": "details['lags used']", "nobs": "details['observations']",
             "critvalues": "critical_values"},
    default_stage="test", stages=("test",))

KPSS = AdapterSpec(
    "statsmodels.tsa.stattools.kpss", "statsmodels", {"the series (missing values dropped)": "x"},
    (Option("variable", None, "the series pymetrix takes from the dataset"), Option("trend", "regression"),
     Option("lags", "nlags", "a number, auto or legacy")),
    outputs={"kpss_stat": "statistic", "p_value": "p_value", "lags": "details['lags used']", "crit": "critical_values",
             "InterpolationWarning": "details['note'] (p-value at the edge of the table)"},
    default_stage="test", stages=("test",))

ENGLE_GRANGER = AdapterSpec(
    "statsmodels.tsa.stattools.coint", "statsmodels", {"the first variable": "y0", "the others": "y1"},
    (Option("variables", None, "the series pymetrix takes from the dataset; rows with a missing value dropped"),
     Option("trend", "trend"), Option("autolag", "autolag"), Option("max_lags", "maxlag"), *_SIMULATED_OPTIONS,
     Option("exog", None, "fixed levels regressors, with simulated critical values only; statsmodels' coint has none")),
    outputs={"coint_t": "statistic", "pvalue": "p_value", "crit_value": "critical_values (1%, 5%, 10%)"},
    default_stage="test", stages=("test",))

FISHER_ADF = AdapterSpec(
    "statsmodels.tsa.stattools.adfuller (for each entity)", "statsmodels", {"each entity's series (missing values dropped)": "x"},
    (Option("variable", None, "the series pymetrix takes from the dataset"),
     Option("lags", "maxlag", "a number: that many lags (autolag=None); aic / bic: autolag"), Option("trend", "regression")),
    outputs={"pvalue": "combined by pymetrix (Maddala-Wu chi-squared, Choi Z and logit)"},
    default_stage="test", stages=("test",))

JOHANSEN = AdapterSpec(
    "statsmodels.tsa.vector_ar.vecm.coint_johansen", "statsmodels", {"the variables": "endog"},
    (Option("variables", None, "the series pymetrix takes from the dataset"), Option("lags", "k_ar_diff"),
     Option("det_order", "det_order", "-1 none, 0 constant, 1 linear trend"),
     Option("signif", None, "picks the tabulated critical values that choose the rank")),
    outputs={"eig": "eigenvalues", "lr1": "table['trace']", "lr2": "table['max-eig']", "cvt": "table (trace critical values)",
             "cvm": "table (max-eig critical values)"},
    default_stage="test", stages=("test",))

VECM_SELECT_ORDER = AdapterSpec(
    "statsmodels.tsa.vector_ar.vecm.select_order", "statsmodels", {"the variables": "data", "exog": "exog", "exog_coint": "exog_coint"},
    (Option("variables", None, "the series pymetrix takes from the dataset"), Option("max_lags", "maxlags"),
     Option("deterministic", "deterministic"), Option("seasons", "seasons")),
    outputs={"ics": "table", "selected_orders": "best"}, default_stage="test", stages=("test",),
    notes="Every order is compared on the same rows: the first max_lags + 1 are presample.")

STL_DECOMPOSITION = AdapterSpec(
    "statsmodels.tsa.seasonal.STL", "statsmodels", {"the series (its unbroken run; logs when multiplicative)": "endog"},
    (Option("seasonal", "seasonal", stage="model"), Option("trend", "trend", stage="model"), Option("robust", "robust", stage="model"),
     *(Option(k, k, stage="model") for k in ("low_pass", "seasonal_deg", "trend_deg", "low_pass_deg", "seasonal_jump", "trend_jump",
                                              "low_pass_jump")),
     Option("inner_iter", "inner_iter", stage="fit"), Option("outer_iter", "outer_iter", stage="fit"),
     Option("log", None, "pymetrix takes logs first, and the exponent of the seasonal component after")),
    outputs={"seasonal": "{name}_seasonal"}, notes="The period is 4 or 12, from the frequency. Recorded in the variables' extra.")

CLASSICAL_DECOMPOSITION = AdapterSpec(
    "statsmodels.tsa.seasonal.seasonal_decompose", "statsmodels", {"the series (its unbroken run)": "x"},
    (Option("log", "model", "multiplicative when logs, else additive"), Option("two_sided", "two_sided"),
     Option("extrapolate_trend", "extrapolate_trend")),
    outputs={"seasonal": "{name}_seasonal"}, stages=("fit",), notes="The period is 4 or 12, from the frequency.")

DYNAMIC_FACTOR_MQ = AdapterSpec(
    "statsmodels.tsa.statespace.dynamic_factor_mq.DynamicFactorMQ", "statsmodels",
    {"the monthly series (from the first month all have data)": "endog", "the quarterly series": "endog_quarterly"},
    (Option("monthly", None, "the monthly series pymetrix takes from the data"), Option("quarterly", None, "the quarterly series"),
     Option("factors", "factors", stage="model"), Option("factor_order", "factor_orders", stage="model"),
     Option("idiosyncratic_ar1", "idiosyncratic_ar1", stage="model"), Option("max_iter", "maxiter")),
    {"fit.disp": (False, "no optimiser output on the console")},
    {"factors.smoothed": "factors", "params": "loadings", "llf": "stats['loglike']", "mle_retvals['iter']": "provenance.method['iterations']",
     "apply": "update (the same parameters on newer data)", "news": "news"},
    stages=("model", "fit", "apply"))

SPECS = {"Quantile": QUANTREG, "ExponentialSmoothing": ETS, "MarkovSwitching": MARKOV, "ARDL": ARDL, "VECM": VECM, "SARIMA": SARIMAX,
         "Theta": THETA, "STLForecast": STL_FORECAST, "ADF": ADF, "KPSS": KPSS, "EngleGranger": ENGLE_GRANGER,
         "FisherADF": FISHER_ADF, "johansen": JOHANSEN, "VECM.select_order": VECM_SELECT_ORDER,
         "STL": STL_DECOMPOSITION, "Classical": CLASSICAL_DECOMPOSITION,
         "DynamicFactor": DYNAMIC_FACTOR_MQ}

NOT_YET: tuple[str, ...] = ()

# statsmodels calls that stay outside the adapter by design (docs/design/adapters.md, Coverage)
OUTSIDE = ("the Kalman filter engine (MLEModel): called inside pymetrix's own optimiser, with no library arguments to pass; "
           "provenance names the engine",
           "interop.from_statsmodels: wraps a result made outside pymetrix, so there is no call to record")


def dynamic_factor_index(result: Any) -> Any:
    """The dates DynamicFactorMQ estimated on. statsmodels keeps them on a private attribute of the model; read here
    only, so a change in statsmodels breaks one place with a clear message."""
    model = result.model
    try:
        return model._index
    except AttributeError:
        from ..errors import PyMetrixError

        raise PyMetrixError("this version of statsmodels no longer provides DynamicFactorMQ's index (model._index); "
                            "report it to pymetrix") from None


def original_endog(model: Any) -> Any:
    """The data a statsmodels model was given, with its labels (``model.data.orig_endog``)."""
    try:
        return model.data.orig_endog
    except AttributeError:
        from ..errors import PyMetrixError

        raise PyMetrixError("this version of statsmodels no longer provides model.data.orig_endog; report it to pymetrix") from None


@renamed_arguments(target="variable")
def pymetrix_name(name: str, variable: str = "") -> str:
    """statsmodels' lag names in pymetrix's expression syntax: ``y.L1`` → ``y[-1]``, ``x.L0`` → ``x``,
    ``D.y.L1`` → ``d(y)[-1]``, ``ar.L2`` → ``<variable>[-2]``; a regime suffix (``[0]``) is kept."""
    import re

    m = re.fullmatch(r"(?:(D)\.)?(.+?)\.L(\d+)((?:\[\d+\])?)", name)
    if not m:
        return name
    diff, base, lag, regime = m.groups()
    if base == "ar" and variable:
        base = variable
    term = f"d({base})" if diff else base
    k = int(lag)
    return f"{term}[-{k}]{regime}" if k else f"{term}{regime}"
