"""Linearised dynamics: block Jacobians (``Jacobians``) and the general-equilibrium Jacobians ``G`` (``HALinear``),
whose impulse responses are pymetrix ``IRF`` objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from pymetrix.errors import SpecificationError
from pymetrix.output import Table

from . import _provenance
from .model import AR1


def _dense(x: Any, T: int) -> np.ndarray:
    """A library Jacobian entry as a T x T array (simple blocks' Jacobians are sparse objects with ``matrix(T)``)."""
    if hasattr(x, "matrix"):
        return np.asarray(x.matrix(T), dtype=float)
    return np.asarray(x, dtype=float)


class Jacobians:
    """T x T Jacobians by output and input: ``J["C", "r"][s, t]`` is the response of C at date s to a change in r at
    date t, at the steady state. ``J.table("C", "r")`` shows a corner; ``J.raw`` is the library's ``JacobianDict``."""

    def __init__(self, matrices: Mapping[tuple[str, str], np.ndarray], outputs: Sequence[str], inputs: Sequence[str],
                 T: int, about: str, provenance: Any = None, raw: Any = None) -> None:
        self.matrices = dict(matrices)
        self.outputs, self.inputs, self.T, self.about = list(outputs), list(inputs), int(T), about
        self.provenance, self.raw = provenance, raw

    @classmethod
    def from_library(cls, J: Any, T: int, model: Any, steady: Any, about: str, calls: dict[str, Any],
                     caught: tuple[str, ...], printed: tuple[str, ...]) -> Jacobians:
        outputs, inputs = [str(o) for o in J.outputs], [str(i) for i in J.inputs]
        matrices = {(o, i): _dense(J[o][i], T) for o in outputs for i in inputs if i in J[o]}
        prov = _provenance.make(model, f"Jacobians of {about}", calls, caught, printed, steady=steady,
                                resolved={"T": int(T), "method": "fake-news algorithm (het blocks); derivatives (simple blocks)"})
        return cls(matrices, outputs, inputs, T, about, prov, J)

    def __getitem__(self, key: tuple[str, str]) -> np.ndarray:
        output, inp = key
        if (output, inp) not in self.matrices:
            raise KeyError(f"no Jacobian of {output!r} with respect to {inp!r}; outputs {self.outputs}, inputs {self.inputs}")
        return self.matrices[output, inp]

    def __repr__(self) -> str:
        return f"<Jacobians of {self.about}: outputs {self.outputs}, inputs {self.inputs}, T={self.T}>"

    def to_frame(self, output: str, input: str) -> pd.DataFrame:  # noqa: A002 - the library's word
        """One Jacobian as a table: a row per date of the output, a column per date of the input."""
        m = self[output, input]
        return pd.DataFrame(m, index=pd.RangeIndex(m.shape[0], name=f"{output} at"),
                            columns=pd.RangeIndex(m.shape[1], name=f"{input} at"))

    def table(self, output: str, input: str, size: int = 5) -> Table:  # noqa: A002
        """The top-left ``size`` x ``size`` corner of one Jacobian."""
        frame = self.to_frame(output, input).iloc[:size, :size]
        frame.columns = [f"t={c}" for c in frame.columns]
        frame.index = [f"s={i}" for i in frame.index]
        return Table.from_frame(frame, title=f"d{output}(s) / d{input}(t), {self.about}",
                                notes=[f"T = {self.T}; the first {min(size, self.T)} dates"])


class HALinear:
    """The model linearised at its steady state: the general-equilibrium Jacobians ``G`` for horizon T, mapping any
    path of the shocks to every output's path (Auclert, Bardóczy, Rognlie and Straub 2021). ``lin.jacobian("K",
    "Z")`` is one of them; ``lin.irf(40)`` gives the responses to each shock's AR(1) path as a pymetrix ``IRF``."""

    def __init__(self, model: Any, steady: Any, G: Any, T: int, calls: dict[str, Any], seconds: float,
                 caught: tuple[str, ...], printed: tuple[str, ...]) -> None:
        self.model, self.steady, self.G, self.T = model, steady, G, int(T)
        self.jacobians = Jacobians.from_library(G, T, model, steady, "the general equilibrium", calls, caught, printed)
        self.provenance = _provenance.make(model, "general-equilibrium Jacobians", calls, caught, printed, steady=steady,
                                           resolved={"T": self.T, "seconds": round(float(seconds), 3)})
        self.jacobians.provenance = self.provenance

    def __repr__(self) -> str:
        return f"<HALinear {self.model.name!r}: G for {self.jacobians.inputs} -> {len(self.jacobians.outputs)} outputs, T={self.T}>"

    @property
    def outputs(self) -> list[str]:
        return list(self.jacobians.outputs)

    def jacobian(self, output: str, shock: str) -> np.ndarray:
        """``G[output][shock]``: the T x T response of ``output`` at each date to ``shock`` at each date."""
        return self.jacobians[output, shock]

    def paths(self, shocks: Mapping[str, Any] | None = None) -> dict[str, dict[str, np.ndarray]]:
        """``{shock: {output: path}}``, each path of length T: ``G`` times the shock's path."""
        given = self.model.shock_paths(shocks, self.T)
        unknown = [k for k in given if k not in self.jacobians.inputs]
        if unknown:
            raise SpecificationError(f"{', '.join(map(repr, unknown))}: G was computed for {self.jacobians.inputs}")
        return {k: {o: self.jacobians[o, k] @ dz for o in self.outputs if (o, k) in self.jacobians.matrices}
                for k, dz in given.items()}

    def irf(self, periods: int = 40, shocks: Mapping[str, Any] | None = None,
            responses: Sequence[str] | None = None) -> Any:
        """Impulse responses, as deviations from the steady state, to each shock's path: its ``AR1`` (``sd *
        rho**t``) or a path given in ``shocks``. A pymetrix ``IRF`` (``values[h, response, shock]``, h = 0 to
        ``periods``): the shocks' own paths first, then every output but the equilibrium targets (zero along the
        path)."""
        from pymetrix.var import IRF

        if int(periods) != periods or not 0 <= periods < self.T:
            raise SpecificationError(f"periods is the last horizon, a whole number from 0 to T-1 = {self.T - 1}, not {periods!r}")
        periods = int(periods)
        given = self.model.shock_paths(shocks, self.T)
        by_shock = self.paths(given)
        names = list(given)
        outputs = [o for o in self.outputs if o not in self.model.targets and o not in names]
        chosen = [*names, *outputs] if responses is None else list(responses)
        unknown = [r for r in chosen if r not in names and r not in outputs]
        if unknown:
            raise SpecificationError(f"no response {', '.join(map(repr, unknown))}; the responses are {', '.join([*names, *outputs])}")
        values = np.zeros((periods + 1, len(chosen), len(names)))
        for k, s in enumerate(names):
            for i, r in enumerate(chosen):
                path = given[s] if r == s else None if r in names else by_shock[s].get(r)  # another shock: zero
                if path is not None:
                    values[:, i, k] = path[: periods + 1]
        described = ", ".join(f"{k}: AR(1) rho={v.rho:g}, sd={v.sd:g}" if isinstance(v, AR1) else f"{k}: a path given"
                              for k, v in (shocks or self.model.shocks).items())
        return IRF(values, chosen, names,
                   identification=f"sequence-space Jacobian at the steady state, first order, T={self.T} ({described})")
