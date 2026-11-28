"""``HATransition``: a perfect-foresight transition after an unexpected shock, as a pymetrix ``Solution`` (a Dataset
of the paths, in levels, with provenance)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from pymetrix.data import Dataset
from pymetrix.errors import SpecificationError
from pymetrix.models.solution import Solution

from . import _provenance


def _frame(impulse: Any, steady: Any, periods: pd.PeriodIndex) -> pd.DataFrame:
    """The library's paths (deviations from the steady state) as levels, a column per aggregate."""
    cols = {}
    for name in impulse:
        dev = np.asarray(impulse[name], dtype=float)
        if dev.ndim == 1 and len(dev) == len(periods):
            cols[str(name)] = steady.values.get(str(name), 0.0) + dev
    return pd.DataFrame(cols, index=pd.PeriodIndex(periods, name="time"))


@dataclass(frozen=True, eq=False, repr=False)
class HATransition(Solution):
    """The transition's paths in levels (a column per aggregate, a row per period from the shock's date), with:

    - ``deviations()``: the same paths less the steady state;
    - ``linear``: the linear transition (a ``Solution``), beside a nonlinear one, for a check on size effects;
    - ``errors``: each Newton iteration's largest absolute error in each target (the last is below the tolerance);
    - ``steady_values``: the steady state the paths start from and return to.
    """

    linear: Any = None
    errors: tuple[dict[str, float], ...] = ()
    steady_values: Any = None
    nonlinear: bool = True

    def __repr__(self) -> str:
        span = f"{self.span[0]} to {self.span[1]}" if self.span else "?"
        kind = f"nonlinear, {self.iterations} Newton iterations" if self.nonlinear else "linear"
        return f"<HATransition {span}, {kind}>"

    def deviations(self) -> Dataset:
        """The paths less the steady state."""
        frame = self.frame.copy()
        for c in frame.columns:
            frame[c] = frame[c] - (self.steady_values or {}).get(c, 0.0)
        return Dataset.from_frame(frame)

    @classmethod
    def build(cls, model: Any, steady: Any, paths: dict[str, np.ndarray], lin: Any, nonlin: Any,
              errors: list[dict[str, float]], calls: dict[str, Any], seconds: float, start: str,
              caught: tuple[str, ...], printed: tuple[str, ...]) -> HATransition:
        T = model.horizon
        try:
            first = pd.Period(str(start))
        except (ValueError, TypeError) as exc:
            raise SpecificationError(f"start is the date of period 0, such as 2000Q1, not {start!r}") from exc
        periods = pd.period_range(first, periods=T, freq=first.freq)
        shocks = {k: (model.shocks[k].to_dict() if k in model.shocks and np.allclose(model.shocks[k].path(T), v)
                      else {"process": "path given", "first": [float(x) for x in v[:8]]}) for k, v in paths.items()}
        sample = (str(periods[0]), str(periods[-1]))
        resolved = {"T": T, "seconds": round(float(seconds), 3), "dates": "labels: period 0 is the shock's date"}
        if nonlin is not None:
            resolved["iterations"] = [{"iteration": i, **e} for i, e in enumerate(errors)]
        prov = _provenance.make(model, "nonlinear perfect-foresight transition" if nonlin is not None else "linear transition",
                                calls, caught, printed, steady=steady, resolved=resolved, spec={"shock_paths": shocks})
        prov = replace(prov, sample=sample, nobs=T)
        lin_frame = _frame(lin, steady, periods)
        lin_data = Dataset.from_frame(lin_frame)
        linear = cls(lin_data.frame, lin_data.axes, lin_data.variables, meta=lin_data.meta, model=model, converged=True,
                     iterations=0, provenance=replace(prov, method={**prov.method, "type": "linear transition"}),
                     steady_values=dict(steady.values), nonlinear=False)
        if nonlin is None:
            return linear
        data = Dataset.from_frame(_frame(nonlin, steady, periods))
        return cls(data.frame, data.axes, data.variables, meta=data.meta, model=model, converged=True,
                   iterations=max(len(errors) - 1, 0), provenance=prov, linear=linear, errors=tuple(errors),
                   steady_values=dict(steady.values), nonlinear=True)

