"""Global sensitivity analysis of linearised models: prior stability mapping (Ratto 2008; Dynare's ``dynare_sensitivity``).

    dyn = read_mod("nk.mod")                       # priors from estimated_params (or a model's parameter specs)
    st = stability_mapping(dyn, draws=2000, seed=1)
    st.shares                                      # determinate, indeterminate, explosive, no steady state, no solution
    st.smirnov                                     # each parameter: determinate draws against the others
    st.plot()                                      # each parameter's two cumulative distributions

**Stability mapping.** Parameters are drawn from their priors (Monte Carlo, from pymetrix's random generator and
``seed``), and the model is solved at each draw. Each draw falls in one class:

- *determinate*: a unique stable solution (the Blanchard-Kahn conditions hold and the solution is found);
- *indeterminate*: more stable roots than predetermined variables (many stable solutions);
- *explosive*: fewer stable roots than predetermined variables (no stable solution);
- *no steady state*: the steady state is not found at these values (or the model cannot be linearised there);
- *no solution*: the roots count right but the stable roots do not determine the predetermined variables (Dynare's
  rank condition), or the solver fails otherwise.

For each parameter, the two-sample Kolmogorov-Smirnov (Smirnov) test compares its distribution in the determinate
draws with its distribution in the others (SciPy's ``ks_2samp``, two-sided). A small p-value means the parameter
drives the model into or out of the determinate region: its prior puts mass where the model has no unique stable
solution. Ratto (2008) reads the statistic, the largest distance between the two cumulative distributions, as the
parameter's importance; Dynare flags p-values below ``pvalue_ks`` (0.001 by default, as here). The same test against
each failing class alone is in ``smirnov_by_class``.

Only the model's own (structural) parameters are drawn: shock sizes, correlations and measurement errors do not
change the solution's existence or uniqueness, so priors on them are left out. Parameters without a prior keep their
calibrated values. Draws are random, not quasi-random: Dynare's default is a Sobol sequence, so shares agree only
within sampling error (binomial: ``sqrt(p (1 - p) / draws)``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .._api import api
from ..output import Table
from .solvers import ModelError

__all__ = ["OUTCOMES", "StabilityMapping", "classify", "stability_mapping"]

OUTCOMES = ("determinate", "indeterminate", "explosive", "no steady state", "no solution")
_UNIT = 1e-6  # as linear._UNIT: a root within this of the unit circle counts as stable (unit roots, as Dynare's qz_criterium)


def _source(source: Any) -> tuple[Any, tuple[str, ...], dict[str, float], dict[str, Any]]:
    """The model, its shocks, the steady-state guess and the priors on its parameters, from a DynareModel, a Template, a
    FittedModel or a Model."""
    from .dynare import DynareModel
    from .model import Model
    from .templates import Template

    if isinstance(source, DynareModel):
        return source.model, tuple(source.exogenous), dict(source.guess), dict(source.priors)
    if isinstance(source, Template):
        model, shocks, guess = source.model, tuple(source.shocks), dict(source.guess)
    elif isinstance(source, Model):
        model, shocks, guess = source, tuple(source.compile().shocks), {}
    elif type(source).__name__ == "FittedModel":
        model, shocks, guess = source.model, tuple(source.shocks), dict(getattr(source, "guess", None) or {})
        priors = dict(getattr(source, "priors", None) or {})
        return model, shocks, guess, {**_spec_priors(model), **priors}
    else:
        raise ModelError(f"a stability mapping needs a model with priors (a Dynare model, a template or a Model), not a {type(source).__name__}")
    return model, shocks, guess, _spec_priors(model)


def _spec_priors(model: Any) -> dict[str, Any]:
    return {p.name: p.prior for p in model.parameters if getattr(p, "prior", None) is not None}


def classify(model: Any, values: Mapping[str, float], shocks: Sequence[str], guess: Mapping[str, float] | None = None) -> str:
    """The class of the model at these parameter values (one of ``OUTCOMES``): found by solving its steady state,
    linearising there and counting the stable roots of the linear model's pencil against its predetermined variables
    (``LinearModel.roots``, the count ``LinearModel.solve`` checks); a determinate count is then solved."""
    try:
        m = model.with_parameters(dict(values)) if values else model
        ss = m.steady_state(guess=dict(guess) if guess else None)
        values_ok = np.all(np.isfinite(np.asarray(list(ss.values.values()), dtype=float)))
        if not values_ok:
            return "no steady state"
        lin = m.linearize(ss, shocks=list(shocks))
    except (ModelError, ValueError, ArithmeticError, np.linalg.LinAlgError):
        return "no steady state"
    try:
        roots, predetermined = lin.roots()
    except (ModelError, ValueError, np.linalg.LinAlgError):
        return "no solution"
    if not np.all(np.isfinite(roots) | np.isinf(roots)):
        return "no solution"
    stable = int(np.sum(np.abs(roots) < 1 + _UNIT))
    if stable > predetermined:
        return "indeterminate"
    if stable < predetermined:
        return "explosive"
    try:
        lin.solve()
    except (ModelError, ValueError, np.linalg.LinAlgError):
        return "no solution"
    return "determinate"


@dataclass(frozen=True, eq=False)
class StabilityMapping:
    """A prior stability mapping: see the module docstring."""

    names: tuple[str, ...]  # the parameters drawn
    draws: pd.DataFrame  # one row per draw: the parameters' values and "outcome"
    priors: Mapping[str, Any] = field(default_factory=dict)
    seed: Any = None
    pvalue_ks: float = 0.001

    @property
    def outcome(self) -> pd.Series:
        return self.draws["outcome"]

    @property
    def counts(self) -> pd.Series:
        """The number of draws in each class (all classes, zeros included)."""
        return self.outcome.value_counts().reindex(list(OUTCOMES), fill_value=0)

    @property
    def shares(self) -> pd.Series:
        """The share of draws in each class."""
        return self.counts / len(self.draws)

    def _test(self, other: Any) -> pd.DataFrame:
        from scipy import stats

        good = self.outcome == "determinate"
        rest = self.outcome.isin(other) if not isinstance(other, str) else self.outcome == other
        rows = []
        for n in self.names:
            a, b = self.draws.loc[good, n].to_numpy(float), self.draws.loc[rest, n].to_numpy(float)
            if len(a) and len(b):
                res = stats.ks_2samp(a, b)
                rows.append([float(res.statistic), float(res.pvalue), len(a), len(b), bool(res.pvalue < self.pvalue_ks)])
            else:
                rows.append([np.nan, np.nan, len(a), len(b), False])
        return pd.DataFrame(rows, index=pd.Index(self.names, name="parameter"),
                            columns=["statistic", "p-value", "determinate", "others", "drives"])

    @property
    def smirnov(self) -> pd.DataFrame:
        """For each parameter: the Smirnov (two-sample Kolmogorov-Smirnov) statistic and p-value of its values in the
        determinate draws against all the others, the two counts, and whether the p-value is below ``pvalue_ks``."""
        return self._test([o for o in OUTCOMES if o != "determinate"])

    def smirnov_by_class(self) -> dict[str, pd.DataFrame]:
        """The same test of the determinate draws against each failing class that has draws (``{class: table}``)."""
        return {o: self._test(o) for o in OUTCOMES[1:] if int(self.counts[o])}

    def summary(self) -> Table:
        counts = self.counts
        rows = [[int(counts[o]), float(counts[o] / len(self.draws)), float(np.sqrt(max(counts[o], 0) * (len(self.draws) - counts[o])
                                                                                    / len(self.draws) ** 3))] for o in OUTCOMES]
        notes = [f"{len(self.draws)} draws from the priors of {', '.join(self.names)} (seed {self.seed}); other parameters at "
                 "their calibration", "s.e.: the binomial standard error of the share, sqrt(p (1 - p) / draws)"]
        return Table(["draws", "share", "s.e."], rows, index=list(OUTCOMES), title="Prior stability mapping", notes=notes)

    def table(self) -> Table:
        """The Smirnov tests, determinate draws against the others."""
        frame = self.smirnov
        rows = [[float(r["statistic"]), float(r["p-value"]), int(r["determinate"]), int(r["others"]),
                 "yes" if r["drives"] else ""] for _, r in frame.iterrows()]
        notes = ["Smirnov (two-sample Kolmogorov-Smirnov, two-sided) test of each parameter's values in the determinate "
                 "draws against those in all the others (SciPy's ks_2samp)",
                 f"drives: p-value below {self.pvalue_ks:g}, the parameter moves the model in or out of the determinate region"]
        return Table(["statistic", "p-value", "determinate", "others", "drives"], rows, index=list(self.names),
                     title="Smirnov tests: determinate against the other draws", notes=notes)

    def plot(self, columns: int = 3) -> Any:
        """Each parameter's cumulative distribution in the determinate draws and in the others, one panel each."""
        from ..figures import PALETTE, Ecdf, Figure, Panel

        good = self.outcome == "determinate"
        panels = []
        for n in self.names:
            layers = []
            for label, mask, colour in (("determinate", good, PALETTE[0]), ("not determinate", ~good, PALETTE[1])):
                values = self.draws.loc[mask, n].reset_index(drop=True)
                if len(values):
                    layers.append(Ecdf(values.rename(label), label, color=colour))
            if layers:
                panels.append(Panel(tuple(layers), title=n, ylabel="share of draws"))
        if not panels:
            raise ModelError("no draws to plot")
        return Figure(tuple(panels), title="Prior stability mapping: cumulative distributions",
                      columns=min(columns, len(panels)))

    def __repr__(self) -> str:
        share = float(self.shares["determinate"])
        return f"<StabilityMapping: {len(self.draws)} draws, {share:.1%} determinate, {len(self.names)} parameters>"


@api("pymetrix:stability_mapping")
def stability_mapping(source: Any, draws: int = 2000, seed: int | None = None, priors: Mapping[str, Any] | None = None,
                      params: Sequence[str] | None = None, pvalue_ks: float = 0.001) -> StabilityMapping:
    """A prior stability mapping of a model (see ``pymetrix.models.sensitivity``): ``draws`` draws of the parameters
    from their priors (a Dynare model's ``estimated_params``, a model's or template's parameter specs, or ``priors``),
    each classified as determinate, indeterminate, explosive, without a steady state or without a solution; and the
    Smirnov test of each parameter, determinate draws against the others. ``params``: the parameters to draw (default:
    every model parameter with a prior)."""
    from .._random import generator

    if int(draws) < 2:
        raise ModelError("draws: at least 2")
    if not 0 < pvalue_ks < 1:
        raise ModelError("pvalue_ks is a probability, between 0 and 1")
    model, shocks, guess, known = _source(source)
    structural = set(model.parameters.names)
    chosen = {**known, **dict(priors or {})}
    names = list(params) if params is not None else [n for n in chosen if n in structural]
    unknown = [n for n in names if n not in structural]
    if unknown:
        raise ModelError(f"{', '.join(unknown)} are not parameters of the model (shock sizes and correlations do not change "
                         "the solution's existence)")
    missing = [n for n in names if n not in chosen]
    if missing:
        raise ModelError(f"no prior for {', '.join(missing)}: give priors={{name: prior}}")
    if not names:
        raise ModelError("no parameter of the model has a prior: give priors={name: prior} (e.g. Uniform(1, 3))")
    if not guess:  # the steady state at the calibration starts every draw's search
        try:
            guess = dict(model.steady_state().values)
        except (ModelError, ValueError, ArithmeticError):
            guess = {}
    rng = generator(seed)
    values = np.array([[float(chosen[n].rvs(rng)) for n in names] for _ in range(int(draws))])
    outcomes = [classify(model, dict(zip(names, row)), shocks, guess) for row in values]
    frame = pd.DataFrame(values, columns=names)
    frame["outcome"] = outcomes
    return StabilityMapping(tuple(names), frame, {n: chosen[n] for n in names}, seed, float(pvalue_ks))
