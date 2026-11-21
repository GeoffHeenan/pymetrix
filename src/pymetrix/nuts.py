"""The No-U-Turn sampler (Hoffman and Gelman 2014) in NumPy, as Stan runs it (Betancourt 2017).

    run = sample(logp_grad, starts, draws=1000, warmup=1000, seed=1)   # starts: (chains, k)
    run.draws                                                           # [chain, draw, parameter]
    run.divergent.sum(axis=1), run.step_size, run.info()                # diagnostics

``logp_grad(x)`` returns the log density (up to a constant) and its gradient at ``x``, an unconstrained vector;
``-inf`` where the density is zero. Hamiltonian Monte Carlo moves along the density's level sets by leapfrog steps
of size ``step_size`` with a mass matrix whose inverse, ``inv_metric``, is the draws' variance: diagonal
(``metric="diag"``, the default) or dense (``metric="dense"``, the draws' whole covariance); NUTS doubles
the trajectory, forwards or backwards at random, until it turns back on itself (the generalised no-U-turn criterion on
the summed momenta, with Stan's extra checks across the two halves of each doubling) or reaches ``max_depth``
doublings. The next draw is chosen from the whole trajectory by its weight ``exp(-H)``: multinomially within each new
half, and biased towards the new half when the two are joined (Betancourt 2017, appendix A). A step whose energy error
exceeds 1000 is a divergence: the trajectory stops, and the draw is flagged, because the sampler cannot follow the
density there (usually a funnel or a sharp curvature) and the draws near it may be biased.

Warmup adapts, as Stan does: the step size by dual averaging towards an average acceptance statistic of
``target_accept`` (0.8; gamma 0.05, t0 10, kappa 0.75), and the inverse metric from the draws of slow windows
of 25, 50, 100, ... iterations after an initial buffer of 75 and before a final buffer of 50 (15%, 75% and 10% of a
warmup shorter than 150), each estimate regularised as Stan's: ``n/(n+5) var + 1e-3 * 5/(n+5)`` (diagonal), or
``n/(n+5) S + 1e-3 * 5/(n+5) I`` with ``S`` the window's sample covariance (dense). ``adapt_metric=False`` keeps the
starting ``inv_metric`` (a dense one from the inverse Hessian at a posterior mode, say) and adapts the step size alone.
Nothing adapts after warmup, so the kept draws are a Markov chain with one fixed kernel.

A dense metric undoes linear correlations: a posterior that is Gaussian with any covariance becomes, in the metric's
coordinates, a standard normal, so a strongly correlated one costs a few leapfrog steps a draw where a diagonal metric
needs ``sqrt(condition number)`` times as many. It costs a ``k x k`` product a step (negligible beside a model's
likelihood) and needs about ``k`` draws in each window to estimate, so the first windows of a long parameter vector
are noisy; started from a good covariance (the mode's) it needs little warmup.

The leapfrog map preserves volume and is reversible for any gradient field, and every draw is chosen by the exact
energy, so a gradient that is only approximate (finite differences) costs efficiency, not correctness.

Each chain has its own random stream spawned from the seed (``pymetrix._random``). R-hat and effective sample sizes
are those of the other samplers (``pymetrix.models.bayes.rhat``, ``ess``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._random import generator as make_rng, seed_sequence
from .errors import SpecificationError

__all__ = ["METRICS", "NUTSRun", "Transform", "sample"]

METRICS = ("diag", "dense")

_MAX_DELTA_H = 1000.0  # an energy error above this is a divergence, as Stan's
_GAMMA, _T0, _KAPPA = 0.05, 10.0, 0.75  # dual averaging, as Stan's


@dataclass(frozen=True, eq=False)
class NUTSRun:
    """The kept draws of a NUTS run, chain by chain, with its diagnostics."""

    draws: np.ndarray  # [chain, draw, parameter]
    logp: np.ndarray  # [chain, draw]: the log density of each draw (unconstrained, as sampled)
    step_size: np.ndarray  # [chain]: the adapted step size
    inv_metric: np.ndarray  # [chain, parameter] (diagonal) or [chain, parameter, parameter] (dense): the adapted inverse metric
    divergent: np.ndarray  # [chain, draw]: whether the draw's trajectory diverged
    depth: np.ndarray  # [chain, draw]: tree depth (doublings)
    leapfrog: np.ndarray  # [chain, draw]: leapfrog steps (gradient evaluations)
    accept: np.ndarray  # [chain, draw]: the acceptance statistic
    warmup_divergences: np.ndarray  # [chain]
    warmup: int
    max_depth: int
    target_accept: float
    metric: str = "diag"

    def info(self) -> dict[str, Any]:
        """Diagnostics as plain values: for provenance and summaries."""
        return {"step size": [round(float(s), 6) for s in self.step_size],
                "divergences": [int(d) for d in self.divergent.sum(axis=1)],
                "warmup divergences": [int(d) for d in self.warmup_divergences],
                "mean tree depth": [round(float(d), 3) for d in self.depth.mean(axis=1)],
                "at max depth": [int(d) for d in (self.depth >= self.max_depth).sum(axis=1)],
                "mean acceptance statistic": [round(float(a), 4) for a in self.accept.mean(axis=1)],
                "gradient evaluations": int(self.leapfrog.sum()),
                "max depth": int(self.max_depth), "target acceptance": float(self.target_accept), "warmup": int(self.warmup),
                "metric": "dense" if self.metric == "dense" else "diagonal"}

    def notes(self) -> list[str]:
        """What a summary says about the run: step sizes, depth, acceptance, and divergences or saturated trees."""
        info = self.info()
        chains, n = self.divergent.shape
        out = [f"No-U-Turn sampler (multinomial, {info['metric']} metric): {chains} chain(s) of {n} draws after {self.warmup} warmup; "
               f"step size {', '.join(f'{s:.3g}' for s in self.step_size)}; mean tree depth "
               f"{', '.join(f'{d:.1f}' for d in info['mean tree depth'])}; acceptance statistic "
               f"{', '.join(f'{a:.2f}' for a in info['mean acceptance statistic'])} (target {self.target_accept:g})"]
        divergent = int(self.divergent.sum())
        if divergent:
            out.insert(0, f"{divergent} divergent transition(s) after warmup: the draws may be biased near them; raise "
                          "target_accept, or reparameterise")
        saturated = sum(info["at max depth"])
        if saturated:
            out.append(f"{saturated} draw(s) reached the maximum tree depth ({self.max_depth}): the trajectories were cut short")
        return out


# --- transformations to an unconstrained space --------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Transform:
    """Each coordinate between its bounds mapped to the real line: logistic between two finite bounds, exponential
    above (or below) one, the identity with none. ``x = to_x(z)``; ``log_jacobian(z)`` is log |dx/dz|."""

    lows: np.ndarray
    highs: np.ndarray

    @classmethod
    def of(cls, bounds: Any) -> Transform:
        b = np.asarray(bounds, dtype=float).reshape(-1, 2)
        return cls(b[:, 0].copy(), b[:, 1].copy())

    def __post_init__(self) -> None:
        lo, hi = np.isfinite(self.lows), np.isfinite(self.highs)
        if np.any(lo & hi & ~(self.highs > self.lows)):
            raise SpecificationError("each upper bound must be above its lower bound")
        object.__setattr__(self, "_masks", (lo & hi, lo & ~hi, ~lo & hi))
        object.__setattr__(self, "_any", bool(np.any(lo | hi)))

    def _kinds(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._masks  # type: ignore[attr-defined]

    def to_x(self, z: np.ndarray) -> np.ndarray:
        if not self._any:  # type: ignore[attr-defined]
            return np.array(z, dtype=float)
        both, low, high = self._kinds()
        x = np.array(z, dtype=float)
        s = 0.5 * (1 + np.tanh(0.5 * z[both]))  # the logistic, without overflow
        x[both] = self.lows[both] + (self.highs[both] - self.lows[both]) * s
        x[low] = self.lows[low] + np.exp(z[low])
        x[high] = self.highs[high] - np.exp(z[high])
        return x

    def to_z(self, x: np.ndarray) -> np.ndarray:
        both, low, high = self._kinds()
        z = np.array(x, dtype=float)
        u = (x[both] - self.lows[both]) / (self.highs[both] - self.lows[both])
        u = np.clip(u, 1e-12, 1 - 1e-12)
        z[both] = np.log(u) - np.log1p(-u)
        z[low] = np.log(np.maximum(x[low] - self.lows[low], 1e-300))
        z[high] = np.log(np.maximum(self.highs[high] - x[high], 1e-300))
        return z

    def dx_dz(self, z: np.ndarray) -> np.ndarray:
        if not self._any:  # type: ignore[attr-defined]
            return np.ones(len(z))
        both, low, high = self._kinds()
        d = np.ones(len(z))
        s = 0.5 * (1 + np.tanh(0.5 * z[both]))
        d[both] = (self.highs[both] - self.lows[both]) * s * (1 - s)
        d[low] = np.exp(z[low])
        d[high] = -np.exp(z[high])
        return d

    def log_jacobian(self, z: np.ndarray) -> float:
        if not self._any:  # type: ignore[attr-defined]
            return 0.0
        both, low, high = self._kinds()
        zb = z[both]
        # log s + log(1 - s) = -softplus(-z) - softplus(z)
        value = float(np.sum(np.log(self.highs[both] - self.lows[both]) - np.logaddexp(0, -zb) - np.logaddexp(0, zb)))
        return value + float(np.sum(z[low])) + float(np.sum(z[high]))

    def wrap(self, f: Callable[[np.ndarray], tuple[float, np.ndarray]]) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
        """``f`` (log density and gradient in x) as a log density and gradient in z, with the Jacobian."""
        if not self._any:  # type: ignore[attr-defined]
            return f
        both, low, high = self._kinds()
        one = low | high
        width = self.highs[both] - self.lows[both]
        log_width = float(np.sum(np.log(width)))

        def in_z(z: np.ndarray) -> tuple[float, np.ndarray]:
            x = np.array(z, dtype=float)
            d = np.ones(len(z))
            gj = np.zeros(len(z))
            zb = z[both]
            s = 0.5 * (1 + np.tanh(0.5 * zb))
            x[both] = self.lows[both] + width * s
            d[both] = width * s * (1 - s)
            gj[both] = 1 - 2 * s
            x[low] = self.lows[low] + np.exp(z[low])
            x[high] = self.highs[high] - np.exp(z[high])
            d[low], d[high] = np.exp(z[low]), -np.exp(z[high])
            gj[one] = 1.0
            lp, g = f(x)
            if not np.isfinite(lp):
                return -np.inf, np.zeros(len(z))
            lj = log_width - float(np.sum(np.logaddexp(0, -zb) + np.logaddexp(0, zb))) + float(np.sum(z[one]))
            return lp + lj, np.asarray(g, dtype=float) * d + gj

        return in_z

    def grad_log_jacobian(self, z: np.ndarray) -> np.ndarray:
        if not self._any:  # type: ignore[attr-defined]
            return np.zeros(len(z))
        both, low, high = self._kinds()
        d = np.zeros(len(z))
        s = 0.5 * (1 + np.tanh(0.5 * z[both]))
        d[both] = 1 - 2 * s
        d[low | high] = 1.0
        return d


# --- the sampler --------------------------------------------------------------------------------------------------

class _Tree:
    """A trajectory (or part of one): its two ends, the draw it proposes, the sum of its momenta, its log weight."""

    __slots__ = ("xm", "pm", "gm", "sm", "xp", "pp", "gp", "sp", "x", "lp", "g", "rho", "logw", "n", "accept", "diverged",
                 "turned")  # s: the end's velocity M^-1 p (p sharp)


def _leaf(x: np.ndarray, p: np.ndarray, g: np.ndarray, lp: float, logw: float, s: np.ndarray) -> _Tree:
    t = _Tree()
    t.xm = t.xp = t.x = x
    t.pm = t.pp = t.rho = p
    t.sm = t.sp = s
    t.gm = t.gp = t.g = g
    t.lp, t.logw, t.n, t.accept, t.diverged, t.turned = lp, logw, 0, 0.0, False, False
    return t


class _Kernel:
    def __init__(self, logp_grad: Callable[[np.ndarray], tuple[float, np.ndarray]], minv: np.ndarray, eps: float,
                 max_depth: int, rng: np.random.Generator) -> None:
        self.f, self.eps, self.max_depth, self.rng = logp_grad, eps, max_depth, rng
        self.minv = minv

    @property
    def minv(self) -> np.ndarray:
        """The inverse metric: a vector (diagonal) or a matrix (dense)."""
        return self._minv

    @minv.setter
    def minv(self, value: np.ndarray) -> None:
        self._minv = value
        if value.ndim == 2:  # p ~ N(0, M), M = minv^-1 = L^-T L^-1 with minv = L L': p = L^-T z
            L = np.linalg.cholesky(value)
            self._root = np.linalg.solve(L.T, np.eye(len(value)))  # L^-T, applied to each momentum draw
        else:
            self._root = np.sqrt(value)

    def velocity(self, p: np.ndarray) -> np.ndarray:
        """M^-1 p: the position's rate of change (p sharp)."""
        m = self._minv
        return m.dot(p) if m.ndim == 2 else m * p

    def momentum(self) -> np.ndarray:
        z = self.rng.standard_normal(len(self._minv))
        return self._root.dot(z) if self._minv.ndim == 2 else z / self._root

    def kinetic(self, p: np.ndarray, s: np.ndarray | None = None) -> float:
        """p' M^-1 p / 2 (``s``: M^-1 p, when known)."""
        if self._minv.ndim == 2:
            return 0.5 * float(p.dot(self.velocity(p) if s is None else s))
        return 0.5 * float(np.sum(self._minv * p * p))

    def leapfrog(self, x: np.ndarray, p: np.ndarray, g: np.ndarray, eps: float
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        p = p + 0.5 * eps * g
        x = x + eps * self.velocity(p)
        lp, g = self.f(x)
        if not (np.isfinite(lp) and np.all(np.isfinite(g))):
            return x, p, np.zeros_like(x), -np.inf
        return x, p + 0.5 * eps * g, np.asarray(g, dtype=float), float(lp)

    def _turned(self, left: _Tree, right: _Tree, rho: np.ndarray) -> bool:
        """The generalised no-U-turn criterion on the joined trajectory, with Stan's two extra checks across the join."""
        if not (left.sm @ rho > 0 and right.sp @ rho > 0):
            return True
        r = left.rho + right.pm
        if not (left.sm @ r > 0 and right.sm @ r > 0):
            return True
        r = left.pp + right.rho
        return not (left.sp @ r > 0 and right.sp @ r > 0)

    def build(self, x: np.ndarray, p: np.ndarray, g: np.ndarray, depth: int, v: int, H0: float) -> _Tree:
        if depth == 0:
            x1, p1, g1, lp1 = self.leapfrog(x, p, g, v * self.eps)
            s1 = self.velocity(p1)
            H = -lp1 + self.kinetic(p1, s1) if np.isfinite(lp1) else np.inf
            if not np.isfinite(H):
                H = np.inf
            t = _leaf(x1, p1, g1, lp1, H0 - H, s1)
            t.n, t.accept = 1, float(np.exp(min(0.0, H0 - H))) if np.isfinite(H) else 0.0
            t.diverged = bool(H - H0 > _MAX_DELTA_H)
            return t
        inner = self.build(x, p, g, depth - 1, v, H0)
        if inner.diverged or inner.turned:
            return inner
        xe, pe, ge = (inner.xp, inner.pp, inner.gp) if v > 0 else (inner.xm, inner.pm, inner.gm)
        outer = self.build(xe, pe, ge, depth - 1, v, H0)
        if outer.diverged or outer.turned:
            outer.n += inner.n
            outer.accept += inner.accept
            return outer
        logw = float(np.logaddexp(inner.logw, outer.logw))
        pick = outer if np.log(self.rng.uniform()) < outer.logw - logw else inner  # multinomial, uniform within
        return self._join(*((inner, outer) if v > 0 else (outer, inner)), pick, logw)

    def _join(self, left: _Tree, right: _Tree, pick: _Tree, logw: float) -> _Tree:
        t = _Tree()
        t.xm, t.pm, t.gm, t.sm = left.xm, left.pm, left.gm, left.sm
        t.xp, t.pp, t.gp, t.sp = right.xp, right.pp, right.gp, right.sp
        t.x, t.lp, t.g = pick.x, pick.lp, pick.g
        t.rho = left.rho + right.rho
        t.logw, t.n, t.accept, t.diverged = logw, left.n + right.n, left.accept + right.accept, False
        t.turned = self._turned(left, right, t.rho)
        return t

    def transition(self, x: np.ndarray, lp: float, g: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, dict[str, Any]]:
        p0 = self.momentum()
        H0 = -lp + self.kinetic(p0)
        tree = _leaf(x, p0, g, lp, 0.0, self.velocity(p0))
        draw = (x, lp, g)
        n = 0
        accept = 0.0
        diverged = False
        depth = 0
        for depth in range(1, self.max_depth + 1):
            v = 1 if self.rng.uniform() < 0.5 else -1
            xe, pe, ge = (tree.xp, tree.pp, tree.gp) if v > 0 else (tree.xm, tree.pm, tree.gm)
            new = self.build(xe, pe, ge, depth - 1, v, H0)
            n += new.n
            accept += new.accept
            if new.diverged:
                diverged = True
                break
            if new.turned:
                break
            if np.log(self.rng.uniform()) < new.logw - tree.logw:  # biased progressive sampling, towards the new half
                draw = (new.x, new.lp, new.g)
            logw = float(np.logaddexp(tree.logw, new.logw))
            tree = self._join(*((tree, new) if v > 0 else (new, tree)), tree, logw)
            if tree.turned:
                break
        return draw[0], draw[1], draw[2], {"accept": accept / max(n, 1), "n": n, "depth": depth, "diverged": diverged}


def _initial_step(kernel: _Kernel, x: np.ndarray, lp: float, g: np.ndarray, eps: float) -> float:
    """Stan's heuristic: double or halve the step until one leapfrog step's acceptance crosses 0.8."""
    direction = 0
    for _ in range(100):
        p = kernel.momentum()
        H0 = -lp + kernel.kinetic(p)
        _, p1, _, lp1 = kernel.leapfrog(x, p, g, eps)
        h = H0 - (-lp1 + kernel.kinetic(p1)) if np.isfinite(lp1) else -np.inf
        if not np.isfinite(h):
            h = -np.inf
        if direction == 0:
            direction = 1 if h > np.log(0.8) else -1
        if (direction == 1 and not h > np.log(0.8)) or (direction == -1 and not h < np.log(0.8)):
            break
        eps = eps * 2.0 if direction == 1 else eps / 2.0
        if not 1e-12 < eps < 1e7:
            break
    return float(np.clip(eps, 1e-12, 1e7))


class _DualAveraging:
    def __init__(self, eps: float, target: float) -> None:
        self.target = target
        self.restart(eps)

    def restart(self, eps: float) -> None:
        self.mu, self.s_bar, self.x_bar, self.count = float(np.log(10 * eps)), 0.0, 0.0, 0

    def update(self, accept: float) -> float:
        self.count += 1
        eta = 1.0 / (self.count + _T0)
        self.s_bar = (1 - eta) * self.s_bar + eta * (self.target - min(1.0, accept))
        x = self.mu - self.s_bar * np.sqrt(self.count) / _GAMMA
        w = self.count ** (-_KAPPA)
        self.x_bar = w * x + (1 - w) * self.x_bar
        return float(np.exp(x))

    def final(self) -> float:
        return float(np.exp(self.x_bar))


def windows(warmup: int) -> tuple[int, list[int]]:
    """Stan's warmup schedule: the end of the initial buffer, and the iterations at which each slow window ends (the
    last stretched to the start of the final buffer). No windows below 20 iterations of warmup."""
    if warmup < 20:
        return warmup, []
    init, term, base = 75, 50, 25
    if init + base + term > warmup:
        init, term = int(0.15 * warmup), int(0.1 * warmup)
        base = warmup - init - term
    ends, start, size = [], init, base
    while True:
        end = start + size
        if end + 2 * size > warmup - term:
            ends.append(warmup - term)
            return init, ends
        ends.append(end)
        start, size = end, 2 * size


def _regularised(window: list[np.ndarray], dense: bool, current: np.ndarray) -> np.ndarray:
    """Stan's estimate of the inverse metric from a slow window's draws: the variances (or the covariance) shrunk
    towards 1e-3 (times the identity), the more so the fewer the draws."""
    n = len(window)
    if n < 2:
        return current
    X = np.array(window)
    w, shrink = n / (n + 5.0), 1e-3 * (5.0 / (n + 5.0))
    if not dense:
        return w * np.var(X, axis=0, ddof=1) + shrink
    S = np.atleast_2d(np.cov(X, rowvar=False))
    return w * 0.5 * (S + S.T) + shrink * np.eye(X.shape[1])


def _chain(logp_grad: Callable[[np.ndarray], tuple[float, np.ndarray]], x0: np.ndarray, draws: int, warmup: int,
           stream: Any, target_accept: float, max_depth: int, inv_metric: np.ndarray | None, adapt_metric: bool,
           progress: Callable[[int], Any] | None, metric: str = "diag") -> tuple:
    rng = make_rng(stream)
    reset = getattr(logp_grad, "reset", None)
    if reset is not None:  # nothing carried over from what ran before in this process: the same draws in any process
        reset()
    x = np.asarray(x0, dtype=float).copy()
    k = len(x)
    lp, g = logp_grad(x)
    if not (np.isfinite(lp) and np.all(np.isfinite(g))):
        raise SpecificationError("the log density or its gradient is not finite where the chain starts")
    g = np.asarray(g, dtype=float)
    dense = metric == "dense"
    if inv_metric is None:
        minv = np.eye(k) if dense else np.ones(k)
    else:
        minv = np.asarray(inv_metric, dtype=float).copy()
        if dense and minv.ndim == 1:
            minv = np.diag(minv)
    kernel = _Kernel(logp_grad, minv, 1.0, max_depth, rng)
    kernel.eps = _initial_step(kernel, x, float(lp), g, 1.0) if warmup > 0 else 1.0
    dual = _DualAveraging(kernel.eps, target_accept)
    init, ends = windows(warmup) if adapt_metric else (warmup, [])
    window: list[np.ndarray] = []
    out = np.empty((draws, k))
    out_lp, acc, depth, steps = np.empty(draws), np.empty(draws), np.empty(draws, dtype=int), np.empty(draws, dtype=int)
    divergent = np.zeros(draws, dtype=bool)
    warm_div = 0
    for i in range(warmup + draws):
        x, lp, g, s = kernel.transition(x, float(lp), g)
        if i < warmup:
            warm_div += s["diverged"]
            kernel.eps = dual.update(s["accept"])
            if ends and init <= i < ends[-1]:
                window.append(x)
                if i + 1 in ends:
                    kernel.minv = _regularised(window, dense, kernel.minv)
                    window = []
                    kernel.eps = _initial_step(kernel, x, float(lp), g, kernel.eps)
                    dual.restart(kernel.eps)
            if i == warmup - 1:
                kernel.eps = dual.final()
        else:
            j = i - warmup
            out[j], out_lp[j], acc[j], depth[j], steps[j], divergent[j] = x, lp, s["accept"], s["depth"], s["n"], s["diverged"]
        if progress is not None and (i + 1) % 50 == 0:
            progress(50)
    return out, out_lp, kernel.eps, kernel.minv, divergent, depth, steps, acc, warm_div


def sample(logp_grad: Callable[[np.ndarray], tuple[float, np.ndarray]], starts: Any, draws: int = 1000, warmup: int = 1000,
           seed: int | None = None, target_accept: float = 0.8, max_depth: int = 10, inv_metric: Any = None,
           adapt_metric: bool = True, progress: Callable[[float], Any] | None = None, workers: int = 1,
           metric: str = "diag") -> NUTSRun:
    """NUTS chains from ``starts`` (one row per chain): ``draws`` kept after ``warmup`` adapting ones each. See the module
    docstring. ``metric``: ``"diag"`` (a diagonal mass matrix, as Stan's default) or ``"dense"``. ``inv_metric``: the
    starting inverse metric (default the identity), e.g. the variances at a mode, or with ``metric="dense"`` their
    covariance (a vector there is a diagonal matrix); ``adapt_metric=False`` keeps it through warmup.
    ``workers``: chains run at once, each in its own process (``pymetrix._processes``), when ``logp_grad`` can be sent
    there (picklable, importable); here otherwise. Each chain has its own random stream, and a ``logp_grad`` with a
    ``reset`` method has it called as each chain starts, so the draws are the same either way."""
    x0 = np.atleast_2d(np.asarray(starts, dtype=float))
    chains, k = x0.shape
    if draws < 1 or warmup < 0 or chains < 1:
        raise SpecificationError("draws must be at least 1, warmup zero or more, and there must be a chain")
    if not 0 < target_accept < 1:
        raise SpecificationError("target_accept is an acceptance rate strictly between 0 and 1 (0.8 as Stan's)")
    if not 1 <= int(max_depth) <= 20:
        raise SpecificationError("max_depth is a number of doublings from 1 to 20")
    if metric not in METRICS:
        raise SpecificationError(f"metric is 'diag' (a diagonal mass matrix) or 'dense', not {metric!r}")
    if inv_metric is not None:
        inv_metric = np.asarray(inv_metric, dtype=float)
        if inv_metric.ndim == 2:
            if metric != "dense":
                raise SpecificationError("a matrix inv_metric is a dense metric: give metric='dense' (or its diagonal)")
            if inv_metric.shape != (k, k) or not np.all(np.isfinite(inv_metric)):
                raise SpecificationError(f"inv_metric must be a finite {k} x {k} matrix, one row per parameter")
            inv_metric = 0.5 * (inv_metric + inv_metric.T)
            try:
                np.linalg.cholesky(inv_metric)
            except np.linalg.LinAlgError:
                raise SpecificationError("inv_metric must be positive definite") from None
        else:
            inv_metric = np.broadcast_to(inv_metric, (k,))
            if not np.all(np.isfinite(inv_metric) & (inv_metric > 0)):
                raise SpecificationError("inv_metric must be positive and finite")
    total = chains * (warmup + draws)
    done = [0]

    def advance(n: int) -> None:
        done[0] += n
        if progress is not None:
            progress(min(1.0, done[0] / total))

    jobs = [(logp_grad, x0[c], int(draws), int(warmup), stream, float(target_accept), int(max_depth), inv_metric, adapt_metric)
            for c, stream in enumerate(seed_sequence(seed).spawn(chains))]
    results = None
    if int(workers) > 1 and chains > 1:
        from ._processes import run_in_processes

        results = run_in_processes([(_chain, (*job, None, metric)) for job in jobs], int(workers),
                                   lambda j, r: advance(int(warmup) + int(draws)), "chain")
    if results is None:
        results = [_chain(*job, advance, metric) for job in jobs]
    parts = list(zip(*results))
    return NUTSRun(np.stack(parts[0]), np.stack(parts[1]), np.array(parts[2]), np.stack(parts[3]), np.stack(parts[4]),
                   np.stack(parts[5]), np.stack(parts[6]), np.stack(parts[7]), np.array(parts[8]), int(warmup), int(max_depth),
                   float(target_accept), metric)
