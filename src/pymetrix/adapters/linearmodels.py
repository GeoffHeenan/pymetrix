"""The linearmodels adapter: IV and panel estimators (``pymetrix.ext.linearmodels``)."""

from __future__ import annotations

from typing import Any

from .base import Adapter, AdapterSpec, Option

ADAPTER = Adapter("linearmodels")


def _clustered(m: Any) -> bool:
    from ..estimation.covariance import Cluster

    return isinstance(getattr(m, "cov", None), Cluster)


_COV = Option("cov", "cov_type", "classic → unadjusted, white → robust, hac → kernel (Bartlett), Cluster(...) → clustered "
              "(with cluster_entity/cluster_time or clusters)")
_HAC = Option("hac_lags", "bandwidth", "with cov='hac'; empty: linearmodels' automatic bandwidth")
_DEBIASED = Option("debiased", "debiased", "small-sample (n-k) correction")
_IV_INPUTS = {"y": "dependent", "exogenous regressors (with const)": "exog", "endog columns": "endog", "instruments": "instruments",
              "weights": "weights"}
_IV_OPTIONS = (Option("endog", None, "chooses the endogenous columns of the design"),
               Option("instruments", None, "expressions evaluated by pymetrix into the instrument matrix"))
_PANEL_INPUTS = {"y": "dependent (entity, timestamp index)", "X": "exog", "weights": "weights"}
_OUTPUTS = {"params": "params", "cov": "params (covariance)", "resids": "resid", "fitted_values": "fitted", "nobs": "stats['nobs']",
            "df_resid": "stats['df_resid']", "rsquared": "stats['r2']", "s2": "stats['se_regression'] (square root)"}

LIML = AdapterSpec(
    "linearmodels.iv.IVLIML", "linearmodels", _IV_INPUTS,
    (*_IV_OPTIONS, _COV, _HAC, _DEBIASED, Option("fuller", "fuller", stage="model"), Option("kappa", "kappa", stage="model")),
    outputs={**_OUTPUTS, "kappa": "stats['kappa']", "first_stage.diagnostics": "extra['first_stage_F']"})

IVGMM = AdapterSpec(
    "linearmodels.iv.IVGMM", "linearmodels", _IV_INPUTS,
    (*_IV_OPTIONS, _COV, _HAC, _DEBIASED, Option("weight_type", "weight_type", stage="model"), Option("iter_limit", "iter_limit"),
     Option("tol", "tol")),
    outputs={**_OUTPUTS, "j_stat": "stats['hansen_j'], stats['hansen_j_pvalue']"})

RANDOM_EFFECTS = AdapterSpec(
    "linearmodels.panel.RandomEffects", "linearmodels", _PANEL_INPUTS,
    (_COV, _DEBIASED, Option("small_sample", "small_sample")),
    outputs={**_OUTPUTS, "theta": "stats['theta_mean']", "variance_decomposition": "extra['variance_decomposition']",
             "rsquared_within/between/overall": "stats['r2_within']..."})

BETWEEN = AdapterSpec(
    "linearmodels.panel.BetweenOLS", "linearmodels", _PANEL_INPUTS, (_COV, _DEBIASED, Option("reweight", "reweight")),
    outputs=_OUTPUTS, notes="One observation per entity: residuals and fitted values are by entity.")

FIRST_DIFFERENCE = AdapterSpec(
    "linearmodels.panel.FirstDifferenceOLS", "linearmodels", _PANEL_INPUTS, (_COV, _DEBIASED), outputs=_OUTPUTS,
    notes="The constant drops out, and the first period of each entity is used only to difference.")

FAMA_MACBETH = AdapterSpec(
    "linearmodels.panel.FamaMacBeth", "linearmodels", _PANEL_INPUTS, (_COV, _HAC, _DEBIASED), outputs=_OUTPUTS,
    unsupported=((_clustered, "clustered covariance is not available; use cov='white' or 'hac'"),))

SPECS = {"LIML": LIML, "IVGMM": IVGMM, "RandomEffects": RANDOM_EFFECTS, "Between": BETWEEN, "FirstDifference": FIRST_DIFFERENCE,
         "FamaMacBeth": FAMA_MACBETH}
