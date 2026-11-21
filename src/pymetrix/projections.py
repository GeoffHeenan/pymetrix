"""Local projections (Jordà 2005): impulse responses from one regression per horizon.

For each horizon h = 0, ..., H::

    y[t+h] = a_h + b_h * shock[t] + (lags of y, the shock and controls) + e[t+h]

and ``b_h`` is the response of ``y`` at horizon h to a unit shock::

    lp = LocalProjection("dlog(gdp)*100", "mp_shock", horizons=12, controls=["infl", "rate"]).fit(data)
    lp.irf                 # by horizon: response, standard error, 68% and 90% bands, observations
    lp.fits.coefficients() # the H+1 regressions side by side

Options:

* ``long_difference``: the response is ``y[t+h] - y[t-1]`` (for log levels).
* ``contemporaneous``: controls dated t, i.e. variables ordered before the shock in a recursive
  (Cholesky) identification. Without them the shock is taken as given (e.g. an external
  high-frequency surprise).
* ``cov``: ``hac`` (Newey-West with h+1 lags, for the overlapping errors) or ``white``
  (heteroskedasticity-robust, with one extra lag of every variable: the lag-augmented
  local projections of Montiel Olea and Plagborg-Møller, 2021).
* ``state``: an indicator (between 0 and 1) dated t-1 for state-dependent responses
  (Ramey and Zubairy, 2018): every coefficient differs by state, and the irf has
  one response per state and a test of their difference.
* ``instruments``: external instruments for the shock (LP-IV; Stock and Watson 2018, Ramey 2016): each horizon's
  regression is estimated by two-stage least squares, the shock instrumented by them and every other regressor by
  itself, with the same standard errors; the table gives each horizon's first-stage F (below 10: weak instruments).
* Panels: entity fixed effects and standard errors clustered by entity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

import numpy as np
import pandas as pd

from pymetrix._lazy import stats

from ._api import api
from .errors import NotFoundError, SpecificationError
from .estimation import IV, OLS, Cluster, Equation
from .expr import BinOp, parse_expression
from .forms import Widget
from .output import Table
from .plugins import register

if TYPE_CHECKING:
    from .data import Dataset
    from .selection import Results

__all__ = ["FittedLocalProjection", "LocalProjection"]


def _term(text: str, lag: int = 0) -> str:
    """An expression (lagged), written so the formula reads it as one term."""
    node = parse_expression(text).shift(lag) if lag else parse_expression(text)
    loose = isinstance(node, BinOp) and node.op not in ("*", "/", "^")  # +, -, comparisons, and/or bind looser than *
    return f"({node})" if loose else str(node)


@dataclass(frozen=True)
class LocalProjection:
    """Impulse responses by local projection: one regression of the response at each horizon on the shock.

    Give the response and the shock (series or expressions such as `100*log(gdp)`), the number of horizons,
    the lags of the response, the shock and the controls, and any controls dated at the time of the shock
    (variables ordered before the shock). Standard errors allow for the overlapping errors of
    multi-horizon regressions. A state indicator gives separate responses in each state.

    The response at horizon h is the shock's coefficient in that regression, and it is causal only if the shock
    is as good as random given the controls: an identified surprise, a recursive ordering through the
    contemporaneous controls (as in a Cholesky VAR), or external instruments. Unlike a VAR, local projections do
    not extrapolate a short-run model to long horizons, so they are more robust to misspecified dynamics, but
    their responses are noisier and often jagged, increasingly so at long horizons, where each regression also
    loses observations at the end of the sample. Read the bands rather than single horizons. With instruments,
    check each horizon's first-stage F in the table (below about 10: weak instruments).
    """

    label: ClassVar[str] = "Local projections"
    topics: ClassVar[tuple[str, ...]] = ("local-projections",)

    response: Annotated[str, Widget("text", help="the response variable, e.g. 100*log(gdp) or infl")]
    shock: Annotated[str, Widget("text", help="the shock, e.g. an identified monetary surprise or the policy rate")]
    horizons: Annotated[int, Widget("number", min=0, max=60, help="last horizon, in periods: one regression for each of 0 to this; long horizons are noisier, with fewer observations")] = 12
    lags: Annotated[int, Widget("number", min=0, max=24, help="lags of the response, the shock and the controls")] = 4
    controls: Annotated[Sequence[str], Widget("text", help="other variables whose lags are controlled for")] = ()
    contemporaneous: Annotated[Sequence[str], Widget("text", help="controls dated at the time of the shock (ordered before it)")] = ()
    long_difference: Annotated[bool, Widget("checkbox", help="response y[t+h] - y[t-1], for variables in log levels")] = False
    cov: Annotated[Literal["hac", "white"], Widget("choice", help="hac: Newey-West with h+1 lags; white: lag-augmented robust")] = "hac"
    state: Annotated[str | None, Widget("text", advanced=True, help="state indicator (0 to 1) for state-dependent responses; dated t-1")] = None
    scale: Annotated[float, Widget("number", min=1e-6, max=1e6, advanced=True, help="size of the shock the responses are scaled to")] = 1.0
    instruments: Annotated[Sequence[str], Widget("text", advanced=True,
                                                 help="external instruments for the shock (LP-IV), e.g. a high-frequency surprise")] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "controls", tuple(self.controls))
        object.__setattr__(self, "contemporaneous", tuple(self.contemporaneous))
        object.__setattr__(self, "instruments", tuple(self.instruments))
        if self.horizons < 0 or self.lags < 0:
            raise SpecificationError("horizons and lags cannot be negative")
        if self.instruments and self.state:
            raise SpecificationError("state-dependent LP-IV is not available yet: give instruments or a state, not both")

    def formula(self, h: int, panel_entity: str | None = None) -> str:
        """The regression at horizon ``h`` (for reading, and to run it yourself)."""
        y = parse_expression(self.response)
        lhs = f"({y.shift(h)}) - ({y.shift(-1)})" if self.long_difference else str(y.shift(h))
        lags = self.lags + (1 if self.cov == "white" else 0)  # lag augmentation
        terms = [_term(self.shock)]
        terms += [_term(c) for c in self.contemporaneous]
        for variable in dict.fromkeys((self.response, self.shock, *self.contemporaneous, *self.controls)):  # all enter lagged
            terms += [_term(variable, -lag) for lag in range(1, lags + 1)]
        if self.state:
            z = _term(self.state, -1)
            interacted = [f"{t}*{z}" for t in terms] + [f"{t}*(1 - {z})" for t in terms]
            terms = [z, f"(1 - {z})"] + interacted + ["- 1"]
        if panel_entity:
            terms.append(f"fe({panel_entity})")
        return f"{lhs} ~ " + " + ".join(terms).replace("+ - 1", "- 1")

    @api("pymetrix:local_projection.fit_smooth")
    def fit_smooth(self, data: Dataset, span: Any = None, penalty: float | None = None, order: int = 2, folds: int = 5,
                   alpha: float = 0.10) -> Any:
        """A smooth local projection (Barnichon and Brownlees 2019): the response a penalised cubic spline in the
        horizon, all horizons estimated together; ``penalty`` by cross-validation unless given (0: the ordinary local
        projection on the periods where every horizon is observed). See ``pymetrix.smooth_lp``."""
        from .smooth_lp import fit_smooth

        return fit_smooth(self, data, span, penalty, order, folds, alpha)

    @api("pymetrix:local_projection.fit")
    def fit(self, data: Dataset, span: Any = None, alpha: float = 0.10) -> FittedLocalProjection:
        """Estimate the regression at every horizon. ``alpha`` sets the wide bands (68% bands are always given)."""
        from .selection import Results

        if data.time_axis is None:
            raise SpecificationError("local projections need a time axis")
        entity = data.entity_axes[0].name if data.kind == "panel" else None
        fits = {}
        shock = Equation(f"__lp__ ~ {_term(self.shock)} - 1").terms[0].label  # the shock's term label, for IV's endog
        for h in range(self.horizons + 1):
            cov: Any = Cluster(entity) if entity else ("hac" if self.cov == "hac" else "white")
            lags = {"hac_lags": h + 1} if cov == "hac" else {}
            method = IV(endog=[shock], instruments=list(self.instruments), cov=cov, **lags) if self.instruments else OLS(cov=cov, **lags)
            fits[f"h={h}"] = Equation(self.formula(h, entity)).fit(data, span=span, method=method)
        results = Results(fits, title=f"response of {self.response} to {self.shock}")
        return FittedLocalProjection(self, results, _irf(self, fits, alpha), alpha)


def _irf(spec: LocalProjection, fits: dict[str, Any], alpha: float) -> pd.DataFrame:
    z68, zw = stats.norm.ppf(0.84), stats.norm.ppf(1 - alpha / 2)
    shock = _term(spec.shock)
    rows = []
    for h, (key, fit) in enumerate(fits.items()):
        names = list(fit.params.names)
        row: dict[Any, Any] = {"horizon": h, "nobs": int(fit.stats["nobs"])}
        if spec.instruments:
            first = fit.extra.get("first_stage_F")
            row["first-stage F"] = float(next(iter(first.values())) if isinstance(first, dict) else first) if first is not None else np.nan
        targets = {"": shock} if not spec.state else {"state": f"{shock}*{_term(spec.state, -1)}", "other": f"{shock}*(1 - {_term(spec.state, -1)})"}
        for regime, term in targets.items():
            i = names.index(_label(fit, term))
            b, se = fit.params.values[i] * spec.scale, fit.params.se[i] * spec.scale
            prefix = f"{regime} " if regime else ""
            row.update({f"{prefix}response": b, f"{prefix}se": se, f"{prefix}lower68": b - z68 * se, f"{prefix}upper68": b + z68 * se,
                        f"{prefix}lower": b - zw * se, f"{prefix}upper": b + zw * se})
        if spec.state:  # do the responses differ by state?
            i, j = names.index(_label(fit, targets["state"])), names.index(_label(fit, targets["other"]))
            diff = fit.params.values[i] - fit.params.values[j]
            var = fit.params.cov[i, i] + fit.params.cov[j, j] - 2 * fit.params.cov[i, j]
            row["difference p-value"] = float(2 * stats.norm.sf(abs(diff) / np.sqrt(var))) if var > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("horizon")


def _label(fit: Any, term: str) -> str:
    """The coefficient name of a term as the equation labels it (``dlog(g)`` is labelled ``log(g) - log(g[-1])``)."""
    target = Equation(f"__lp__ ~ {term} - 1").terms[0].label  # parsed as the fitted equation parses its terms
    if target in fit.params.names:
        return target
    raise NotFoundError(f"no coefficient for {term!r} among {list(fit.params.names)}")


@dataclass(frozen=True, eq=False)
class FittedLocalProjection:
    """Local-projection impulse responses: ``irf`` (by horizon) and the regressions behind them (``fits``)."""

    spec: LocalProjection
    fits: Results = field(repr=False)
    irf: pd.DataFrame
    alpha: float

    @property
    def provenance(self) -> Any:
        """The provenance of the first horizon's regression (all use the same data), as a local projection."""
        from dataclasses import replace

        first = next((f for f in self.fits.values() if getattr(f, "provenance", None) is not None), None) if hasattr(self.fits, "values") else None
        if first is None:
            return None
        return replace(first.provenance, method={**(first.provenance.method or {}), "type": "local projection",
                                                 "horizons": len(self.irf)})

    def table(self) -> Table:
        cols = [c for c in self.irf.columns if c != "nobs"] + ["nobs"]
        return Table.from_frame(self.irf[cols], title=f"Local projections: response of {self.spec.response} to {self.spec.shock}",
                                notes=[f"Bands: 68% and {1 - self.alpha:.0%}. Standard errors: "
                                       + ("clustered by entity" if "fe(" in self.spec_formula(0) else
                                          "Newey-West with h+1 lags" if self.spec.cov == "hac" else "robust, lag-augmented")
                                       + (f". Responses scaled to a shock of {self.spec.scale:g}." if self.spec.scale != 1 else "")])

    def spec_formula(self, h: int) -> str:
        fit = self.fits[f"h={h}"]
        return fit.equation.text

    def summary(self) -> Table:
        return self.table()

    def plot(self) -> Any:
        """The responses with 68% and wide bands (one panel per state when state-dependent)."""
        from .figures import local_projection_figure

        return local_projection_figure(self)

    def __str__(self) -> str:
        return str(self.table())

    def _repr_html_(self) -> str:
        return self.table()._repr_html_()


register("models", LocalProjection)
