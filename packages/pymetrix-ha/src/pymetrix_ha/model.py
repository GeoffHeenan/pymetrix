"""``HAModel``: a heterogeneous-agent model as blocks, its calibration, its steady-state and equilibrium problems and
its shocks, solved by sequence-jacobian (Auclert, Bardóczy, Rognlie and Straub 2021).

    from pymetrix_ha import HAModel, AR1
    ks = HAModel([household, firm, mkt_clearing], calibration={...},
                 steady_state={"unknowns": {"beta": 0.98, "Z": 0.85, "K": 3.0}, "targets": {"r": 0.01, "Y": 1.0, "asset_mkt": 0.0}},
                 equilibrium={"unknowns": ["K"], "targets": ["asset_mkt"]}, shocks={"Z": AR1(rho=0.8, sd=0.01)})
    ss = ks.steady_state()          # HASteadyState: ss.table(), ss["K"]
    lin = ks.linearize(ss)          # HALinear: the general-equilibrium Jacobians G; lin.irf(40) is a pymetrix IRF
    path = ks.solve(steady=ss)      # HATransition (a Solution): the nonlinear perfect-foresight transition

Every tolerance and iteration limit is passed to the library from ``HAOptions`` (its values are sequence-jacobian
1.0.0's defaults), and the library's printing is kept in provenance instead of shown.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from pymetrix.errors import SpecificationError

from . import blocks as _blocks
from ._library import library, quiet

__all__ = ["AR1", "HAModel", "HAOptions"]


@dataclass(frozen=True)
class AR1:
    """A shock's path: ``sd * rho**t`` for t = 0, 1, ..., T-1, as deviations from the steady state (an unexpected
    shock of size ``sd`` at date 0 that decays at rate ``rho``)."""

    rho: float
    sd: float = 1.0

    def __post_init__(self) -> None:
        if not -1 < float(self.rho) < 1:
            raise SpecificationError(f"AR1: rho is between -1 and 1, not {self.rho!r}")

    def path(self, T: int) -> np.ndarray:
        return float(self.sd) * float(self.rho) ** np.arange(T)

    def to_dict(self) -> dict[str, Any]:
        return {"process": "AR(1)", "rho": float(self.rho), "sd": float(self.sd)}


@dataclass(frozen=True)
class HAOptions:
    """Every setting pymetrix passes to sequence-jacobian. The defaults are the library's own (1.0.0), passed
    explicitly so that a new release of the library cannot change a result silently.

    - Steady state: ``solver`` (``""``: Brent's method, ``brentq``, for one unknown given as bounds; else Powell's
      hybrid method, ``hybr``, as the Krusell-Smith notebook uses), ``ttol`` (the targets), ``ctol`` (the
      consistency of a constrained solve), ``backward_tol`` and ``backward_maxit`` (the household's policies, by the
      endogenous grid method), ``forward_tol`` and ``forward_maxit`` (the distribution, by Young's method).
    - Jacobians: ``h``, the step of the numerical derivatives inside the household block, one-sided unless
      ``twosided``.
    - Transitions: ``tol``, the largest absolute error in any target, and ``maxit`` Newton iterations.
    """

    solver: str = ""
    ttol: float = 1e-12
    ctol: float = 1e-9
    backward_tol: float = 1e-8
    backward_maxit: int = 5000
    forward_tol: float = 1e-10
    forward_maxit: int = 100_000
    h: float = 1e-4
    twosided: bool = False
    tol: float = 1e-8
    maxit: int = 30

    def steady_solver(self, unknowns: Mapping[str, Any]) -> str:
        if self.solver:
            return self.solver
        bounded = len(unknowns) == 1 and isinstance(next(iter(unknowns.values())), tuple)
        return "brentq" if bounded else "hybr"


def _floats(values: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in values.items():
        if isinstance(v, tuple):
            out[str(k)] = [float(x) for x in v]
        elif isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
            out[str(k)] = float(v) if isinstance(v, (float, np.floating)) else int(v)
        else:
            out[str(k)] = v if isinstance(v, (str, bool, type(None))) else repr(v)
    return out


class HAModel:
    """A heterogeneous-agent model: sequence-jacobian blocks (``@sj.simple``, ``@sj.het``, or ``pymetrix_ha.simple``
    and ``het``, which check them), with

    - ``calibration``: the parameters and grid sizes, by name;
    - ``steady_state``: ``{"unknowns": {name: start, or (low, high) bounds}, "targets": {name: value}}``, the
      calibration problem (``"solver"`` too, to override ``options.solver``);
    - ``equilibrium``: ``{"unknowns": [...], "targets": [...]}`` for the dynamics (the targets are zero along a path);
    - ``shocks``: ``{name: AR1(rho, sd)}``, the exogenous inputs whose responses are asked for;
    - ``horizon``: T, the length of the sequences (the Jacobians are T x T);
    - ``options``: ``HAOptions``, every tolerance and iteration limit.

    The blocks are checked when the model is made: their outputs are names, and every input is calibrated, an
    unknown, a shock or another block's output.
    """

    def __init__(self, blocks: Sequence[Any], calibration: Mapping[str, Any], steady_state: Mapping[str, Any],
                 equilibrium: Mapping[str, Any], shocks: Mapping[str, AR1] | None = None, horizon: int = 300,
                 name: str = "", options: HAOptions | None = None) -> None:
        sj = library()
        self.blocks = list(blocks)
        if not self.blocks:
            raise SpecificationError("HAModel: give at least one block")
        _blocks.check_blocks(self.blocks)
        self.calibration = dict(calibration)
        ss = dict(steady_state)
        unknown_keys = set(ss) - {"unknowns", "targets", "solver"}
        if unknown_keys or "unknowns" not in ss or "targets" not in ss:
            raise SpecificationError('steady_state is {"unknowns": {...}, "targets": {...}} (and "solver", optionally)'
                                     + (f", not {sorted(unknown_keys)}" if unknown_keys else ""))
        self.ss_unknowns = dict(ss["unknowns"])
        self.ss_targets = dict(ss["targets"])
        self.options = options or HAOptions()
        if ss.get("solver"):
            self.options = HAOptions(**{**asdict(self.options), "solver": str(ss["solver"])})
        eq = dict(equilibrium)
        if set(eq) != {"unknowns", "targets"}:
            raise SpecificationError('equilibrium is {"unknowns": [...], "targets": [...]}')
        self.unknowns = [eq["unknowns"]] if isinstance(eq["unknowns"], str) else list(eq["unknowns"])
        self.targets = [eq["targets"]] if isinstance(eq["targets"], str) else list(eq["targets"])
        if len(self.unknowns) != len(self.targets):
            raise SpecificationError(f"equilibrium: as many unknowns as targets, not {len(self.unknowns)} and {len(self.targets)}")
        self.shocks = dict(shocks or {})
        for k, v in self.shocks.items():
            if not isinstance(v, AR1):
                raise SpecificationError(f"shocks: {k!r} is an AR1(rho, sd), not {v!r}")
        if int(horizon) != horizon or horizon < 2:
            raise SpecificationError(f"horizon is T, a whole number of periods (300, say), not {horizon!r}")
        self.horizon = int(horizon)
        _blocks.check_inputs(self.blocks, [*self.calibration, *self.ss_unknowns, *self.unknowns, *self.shocks])
        self.name = name or "HA model"
        self.model = sj.create_model(self.blocks, name=self.name)
        made = set(map(str, self.model.outputs)) | set(map(str, self.model.inputs))
        for what, names in (("steady-state target", self.ss_targets), ("equilibrium target", self.targets),
                            ("equilibrium unknown", self.unknowns), ("shock", self.shocks)):
            missing = [n for n in names if n not in made]
            if missing:
                raise SpecificationError(f"{what} {', '.join(map(repr, missing))} is not an input or output of any block")

    def __repr__(self) -> str:
        return (f"<HAModel {self.name!r}: {len(self.blocks)} blocks, unknowns {self.unknowns}, targets {self.targets}, "
                f"shocks {list(self.shocks)}, T={self.horizon}>")

    # what the library is given ------------------------------------------------------------------------------------

    def _het_options(self, method: str) -> dict[str, dict[str, Any]]:
        """The options each het block takes for ``method``, keyed by its name (the library's per-block options)."""
        o = self.options
        per = {"steady_state": {"backward_tol": o.backward_tol, "backward_maxit": o.backward_maxit,
                                "forward_tol": o.forward_tol, "forward_maxit": o.forward_maxit},
               "jacobian": {"h": o.h, "twosided": o.twosided}}[method]
        return {str(b.name): dict(per) for b in _blocks.het_blocks(self.blocks)}

    def spec(self) -> dict[str, Any]:
        """The model as provenance records it: calibration, blocks (with a fingerprint of their source), the
        steady-state and equilibrium problems, the shocks and the horizon."""
        return {"name": self.name, "calibration": _floats(self.calibration),
                "blocks": [_blocks.describe(leaf) for b in self.blocks for leaf in _blocks.leaves(b)],
                "steady_state": {"unknowns": _floats(self.ss_unknowns), "targets": _floats(self.ss_targets)},
                "equilibrium": {"unknowns": list(self.unknowns), "targets": list(self.targets)},
                "shocks": {k: v.to_dict() for k, v in self.shocks.items()}, "horizon": self.horizon}

    @contextmanager
    def _counting(self, method: str) -> Iterator[list[Any]]:
        """Each call of the combined model's ``method`` inside the block, recorded (the library's steady-state
        solver and Newton loop return no count of their own): the wrapper passes the call through unchanged."""
        calls: list[Any] = []
        original = getattr(self.model, method)

        def counted(*args: Any, **kwargs: Any) -> Any:
            out = original(*args, **kwargs)
            calls.append(out)
            return out

        setattr(self.model, method, counted)
        try:
            yield calls
        finally:
            delattr(self.model, method)

    # the verbs ----------------------------------------------------------------------------------------------------

    def steady_state(self) -> Any:
        """The calibration solved: the unknowns found, the targets met, the household's policies and distribution
        (``HASteadyState``)."""
        from pymetrix._capture import capture
        from pymetrix.threads import blas_threads

        from .steady import HASteadyState

        o = self.options
        solver = o.steady_solver(self.ss_unknowns)
        het = self._het_options("steady_state")
        call = {"solver": solver, "ttol": o.ttol, "ctol": o.ctol, "verbose": False}
        start = time.perf_counter()
        with capture() as caught, quiet() as printed, blas_threads(), self._counting("steady_state") as calls:
            ss = self.model.solve_steady_state(self.calibration, self.ss_unknowns, self.ss_targets,
                                               options=het, **call)
        seconds = time.perf_counter() - start
        return HASteadyState(self, ss, {"solve_steady_state": {**call, "options": het}}, len(calls), seconds,
                             tuple(caught), tuple(printed))

    def _steady(self, steady: Any) -> Any:
        return self.steady_state() if steady is None else steady

    def jacobian(self, block: str | Any, inputs: Sequence[str], outputs: Sequence[str] | None = None,
                 T: int | None = None, steady: Any = None) -> Any:
        """One block's Jacobians at the steady state (``Jacobians``): for a het block, by the fake-news algorithm;
        ``ks.jacobian("household", ["r", "w"], T=5)["C", "r"]`` is the T x T response of consumption at each date
        to the interest rate at each date."""
        from pymetrix._capture import capture
        from pymetrix.threads import blas_threads

        from .linear import Jacobians

        ss = self._steady(steady)
        b = _blocks.find(self.blocks, block)
        T = self.horizon if T is None else int(T)
        het = self._het_options("jacobian").get(str(b.name), {})
        with capture() as caught, quiet() as printed, blas_threads():
            J = b.jacobian(ss.raw, inputs=list(inputs), outputs=None if outputs is None else list(outputs), T=T, **het)
        return Jacobians.from_library(J, T, self, ss, f"the block {b.name!r}", {"jacobian": {"block": str(b.name), "T": T, **het}},
                                      tuple(caught), tuple(printed))

    def linearize(self, steady: Any = None, T: int | None = None, outputs: Sequence[str] | None = None) -> Any:
        """The general-equilibrium Jacobians ``G`` at the steady state, for horizon T: every output's response at
        each date to each shock at each date (``HALinear``)."""
        from pymetrix._capture import capture
        from pymetrix.threads import blas_threads

        from .linear import HALinear

        if not self.shocks:
            raise SpecificationError("linearize: the model has no shocks; give shocks={name: AR1(rho, sd)}")
        ss = self._steady(steady)
        T = self.horizon if T is None else int(T)
        het = self._het_options("jacobian")
        start = time.perf_counter()
        with capture() as caught, quiet() as printed, blas_threads():
            G = self.model.solve_jacobian(ss.raw, self.unknowns, self.targets, list(self.shocks),
                                          outputs=None if outputs is None else list(outputs), T=T, options=het)
        seconds = time.perf_counter() - start
        call = {"solve_jacobian": {"unknowns": self.unknowns, "targets": self.targets, "inputs": list(self.shocks), "T": T,
                                   "options": het}}
        return HALinear(self, ss, G, T, call, seconds, tuple(caught), tuple(printed))

    def shock_paths(self, shocks: Mapping[str, Any] | None, T: int) -> dict[str, np.ndarray]:
        """``{name: path}`` of length T: an ``AR1``'s path, or a path given (deviations from the steady state,
        padded with zeros to T)."""
        chosen = dict(self.shocks) if shocks is None else dict(shocks)
        if not chosen:
            raise SpecificationError("no shocks: give shocks={name: AR1(rho, sd) or a path}")
        out = {}
        for k, v in chosen.items():
            if isinstance(v, AR1):
                out[k] = v.path(T)
                continue
            path = np.asarray(v, dtype=float).ravel()
            if len(path) > T:
                raise SpecificationError(f"the path of {k!r} has {len(path)} periods, more than the horizon T={T}")
            out[k] = np.concatenate([path, np.zeros(T - len(path))])
        return out

    def solve(self, shocks: Mapping[str, Any] | None = None, nonlinear: bool = True, steady: Any = None,
              linear: Any = None, start: str = "2000Q1") -> Any:
        """The perfect-foresight transition after an unexpected shock at date 0 (an "MIT shock"), back to the steady
        state (``HATransition``, a ``Solution``). ``shocks``: ``{name: AR1 or path}`` (the model's shocks by default);
        ``nonlinear=False`` gives the linear transition; ``start`` dates period 0 (the dates are labels: the model
        has no calendar). With ``nonlinear=True`` the linear transition is kept beside it, for comparison."""
        from pymetrix._capture import capture
        from pymetrix.threads import blas_threads

        from .nonlinear import HATransition

        ss = self._steady(steady)
        T = self.horizon
        paths = self.shock_paths(shocks, T)
        unknown = [k for k in paths if k not in set(map(str, self.model.inputs))]
        if unknown:
            raise SpecificationError(f"{', '.join(map(repr, unknown))} is not an input of the model")
        o = self.options
        jac = self._het_options("jacobian")
        start_time = time.perf_counter()
        errors: list[dict[str, float]] = []
        with capture() as caught, quiet() as printed, blas_threads():
            lin = self.model.solve_impulse_linear(ss.raw, self.unknowns, self.targets, dict(paths), options=jac)
            nonlin = None
            if nonlinear:
                call = {"tol": o.tol, "maxit": o.maxit, "verbose": False}
                with self._counting("impulse_nonlinear") as evaluations:
                    try:
                        nonlin = self.model.solve_impulse_nonlinear(ss.raw, self.unknowns, self.targets, dict(paths),
                                                                    options=jac, **call)
                    except ValueError as exc:
                        if "No convergence" in str(exc):
                            raise SpecificationError(f"the transition did not converge in {o.maxit} Newton iterations "
                                                     f"(tol {o.tol:g}); a smaller shock, or a larger maxit in HAOptions") from exc
                        raise
                errors = [{k: float(np.max(np.abs(r[k]))) for k in self.targets} for r in evaluations]
        seconds = time.perf_counter() - start_time
        calls = {"solve_impulse_linear": {"unknowns": self.unknowns, "targets": self.targets, "options": jac}}
        if nonlinear:
            calls["solve_impulse_nonlinear"] = {"unknowns": self.unknowns, "targets": self.targets, "options": jac,
                                                "tol": o.tol, "maxit": o.maxit, "verbose": False}
        return HATransition.build(self, ss, paths, lin, nonlin, errors, calls, seconds, start, tuple(caught), tuple(printed))
