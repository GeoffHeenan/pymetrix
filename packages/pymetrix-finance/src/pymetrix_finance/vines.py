"""Vine copulas: the dependence of several returns built from bivariate copulas, a family for each pair.

    vine = vine_copula(data, ["a", "b", "c", "d"])                    # an R-vine, structure and families selected
    vine.summary()                                                    # tree, edge, family, parameters, Kendall's tau
    vine_copula(data, ["a", "b", "c"], structure="dvine", order=["b", "a", "c"], method="joint")
    vine.portfolio_var([0.25] * 4, level=0.99)                        # VaR and ES by simulation through the margins
    vine_copula(data, ["a", "b", "c"], families="all", gof=True)     # BB6, BB8, Tawn too; Rosenblatt-transform test

**The construction** (Bedford and Cooke 2002; Aas, Czado, Frigessi and Bakken 2009). A regular vine on d variables is
a sequence of trees: the first joins the variables, each later one joins the edges of the tree before it, two of
them only where they share a node (the proximity condition). An edge of tree t joins two variables a and b given a
set D of t - 1 others; it carries a bivariate copula c_{a,b|D} of the conditional distributions F(a | D) and
F(b | D). The density is the product over the edges,

    c(u) = prod_edges c_{a,b|D}(F(a | D), F(b | D)),

and the conditional distributions of the next tree come from the h-functions of this one:
F(a | D, b) = dC_{a,b|D}(x, y) / dy at x = F(a | D), y = F(b | D), and F(b | D, a) = dC / dx. A C-vine has a root in
each tree joined to every other node (a star); a D-vine's trees are paths, the first one an ordering of the
variables. Each pair copula is any of the plugin's bivariate families: the Gaussian, t, Clayton, Gumbel, Frank, Joe,
BB1, BB7, the survival (180 degree) and 90 and 270 degree rotations of Clayton, Gumbel and Joe, or independence (the
default set, ``DEFAULT_PAIR_FAMILIES``); with ``families="all"`` or listed, also BB6, BB8, Tawn's two types and the
rotations of BB1, BB6, BB7, BB8 and Tawn (Tawn's copula is not exchangeable: its transpose is the other type).

**Selection** (Dissmann, Brechmann, Czado and Kurowicka 2013): each tree is the maximum spanning tree of the
absolute empirical Kendall's tau between the conditional pseudo-observations of the candidate edges (C-vines: the
root with the largest sum of |tau|; D-vines: the order that maximises the sum of |tau| along the path, exactly by
dynamic programming up to 12 series, otherwise greedily). Each edge's family is the one of lowest AIC (or BIC) among
those that fit the sign of its Kendall's tau (independence included), optionally after an independence test
(Genest and Favre 2007: the copula is independence when |tau| is not significant at ``level``). Above a truncation
level every pair copula is independence and the trees are any that satisfy the proximity condition.

**Estimation** is sequential, tree by tree: each pair copula by maximum likelihood on the pseudo-observations of
its tree (the Gaussian and t by full maximum likelihood of the correlation and the degrees of freedom), then the
h-functions give the next tree's. ``method="joint"`` refines all the parameters together by maximum likelihood from
the sequential values. The standard errors are each pair's observed information with its inputs taken as known
(they ignore the earlier trees' estimation and the margins').

**Simulation** inverts the Rosenblatt transform: the variables in the reverse order of the vine's matrix (Dissmann et
al.'s lower-triangular R-vine matrix, as in VineCopula), each from a uniform through the inverse h-functions of its
column. The matrix (``vine.matrix()``), with families and parameters in VineCopula's codes, is ``vine.to_vinecopula()``.

**Goodness of fit** (``gof=True``, ``vine.goodness_of_fit()``): the Rosenblatt transform (``rosenblatt``: each variable's
distribution given those below it in its matrix column, from the h-functions) leaves independent uniforms if the vine
is right; the statistic is VineCopula's ECP2 Cramer-von Mises distance between the transformed data's empirical
copula and independence, sum_t (C_n(Z_t) - prod_j Z_tj)^2 (Schepsmeier 2019). The p-value is a parametric bootstrap
that refits the vine: samples from the fitted vine, made pseudo-observations, their parameters re-estimated on the
same structure and families (the selection is not repeated). VineCopula's own ECP2 p-value resamples the data with
the parameters fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np
import pandas as pd
from scipy import optimize
from scipy import stats as sps

from pymetrix.errors import SpecificationError
from pymetrix.extension import api, plugin_context, random_number_generator
from pymetrix.output import Table
from . import PLUGIN
from .copulas import (ARCH, EXTRA_2, MARGINS, ROTATABLE_2, Margin, _prepare, _quiet, _se_from, copula_h, copula_hinv,
                      copula_tau, fit_copula, pseudo_observations, rotated_name, t_logpdf, tail_dependence)

if TYPE_CHECKING:
    from pymetrix.data import Dataset
    from pymetrix.figures import Figure

_ROTATED_1 = tuple(f"{f}{a}" for a in (180, 90, 270) for f in ("clayton", "gumbel", "joe"))
_ROTATED_2 = tuple(rotated_name(f, a) for a in (180, 90, 270) for f in ROTATABLE_2)
ROTATED = _ROTATED_1 + _ROTATED_2
# the families selected among by default (as before BB6, BB8, Tawn and the rotated two-parameter families were
# added: those are in PAIR_FAMILIES, chosen with families=... or families="all")
DEFAULT_PAIR_FAMILIES = ("indep", "gaussian", "t", "clayton", "gumbel", "frank", "joe", "bb1", "bb7", *_ROTATED_1)
PAIR_FAMILIES = DEFAULT_PAIR_FAMILIES + EXTRA_2 + _ROTATED_2
_POSITIVE = ("gaussian", "t", "clayton", "gumbel", "frank", "joe", "bb1", "bb7", "clayton180", "gumbel180", "joe180",
             *EXTRA_2, *(rotated_name(f, 180) for f in ROTATABLE_2))
_NEGATIVE = ("gaussian", "t", "frank", "clayton90", "gumbel90", "joe90", "clayton270", "gumbel270", "joe270",
             *(rotated_name(f, a) for a in (90, 270) for f in ROTATABLE_2))
STRUCTURES = ("rvine", "cvine", "dvine")
# VineCopula's family codes (the 90 and 270 degree rotations there have negative parameters, Tawn's psi excepted)
VC_CODE = {"indep": 0, "gaussian": 1, "t": 2, "clayton": 3, "gumbel": 4, "frank": 5, "joe": 6, "bb1": 7, "bb6": 8,
           "bb7": 9, "bb8": 10, "clayton180": 13, "gumbel180": 14, "joe180": 16, "clayton90": 23, "gumbel90": 24,
           "joe90": 26, "clayton270": 33, "gumbel270": 34, "joe270": 36,
           **{rotated_name(f, a): c + {180: 10, 90: 20, 270: 30}[a] for f, c in (("bb1", 7), ("bb6", 8), ("bb7", 9), ("bb8", 10))
              for a in (180, 90, 270)},
           **{f"tawn{k}": 100 * k + 4 for k in (1, 2)},
           **{f"tawn{k}_{a}": 100 * k + {180: 14, 90: 24, 270: 34}[a] for k in (1, 2) for a in (180, 90, 270)}}
VC_FAMILY = {v: k for k, v in VC_CODE.items()}
_UMIN = 1e-12


# -- pair copulas ------------------------------------------------------------------------------------------------------

def n_pair_parameters(family: str) -> int:
    if family == "indep":
        return 0
    if family == "gaussian":
        return 1
    if family == "t":
        return 2
    return len(ARCH[family].names)


def pair_names(family: str) -> tuple[str, ...]:
    return {"indep": (), "gaussian": ("rho",), "t": ("rho", "nu")}.get(family) or ARCH[family].names


def _pdict(family: str, p: Sequence[float]) -> dict[str, Any]:
    if family in ("gaussian", "t"):
        R = np.array([[1.0, p[0]], [p[0], 1.0]])
        return {"R": R} if family == "gaussian" else {"R": R, "nu": float(p[1])}
    return dict(zip(ARCH[family].names, map(float, p)))


def transpose(family: str) -> str:
    """The family of C(v, u): the 90 and 270 degree rotations swap, and Tawn's two types; the others are
    exchangeable."""
    if family.startswith("tawn"):
        family = ("tawn2" if family[4] == "1" else "tawn1") + family[5:]
    if family.endswith("90") and not family.endswith("270"):
        return family[:-2] + "270"
    if family.endswith("270"):
        return family[:-3] + "90"
    return family


def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, _UMIN, 1 - _UMIN)


@_quiet
def pair_logpdf(family: str, p: Sequence[float], u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
    """log c(u1, u2) of a pair copula."""
    if family == "indep":
        return np.zeros(len(u1))
    if family == "gaussian":
        r = p[0]
        x, y = sps.norm.ppf(u1), sps.norm.ppf(u2)
        return -0.5 * np.log1p(-r * r) - (r * r * (x * x + y * y) - 2 * r * x * y) / (2 * (1 - r * r))
    if family == "t":
        return t_logpdf(np.column_stack([u1, u2]), np.array([[1.0, p[0]], [p[0], 1.0]]), p[1])
    return ARCH[family].logpdf(np.column_stack([u1, u2]), tuple(p))


def pair_h1(family: str, p: Sequence[float], u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
    """P(U2 <= u2 | U1 = u1) = dC/du1 (VineCopula's hfunc1)."""
    if family == "indep":
        return _clip(np.asarray(u2, float))
    return _clip(copula_h(family, _pdict(family, p), u1, u2))


def pair_h2(family: str, p: Sequence[float], u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
    """P(U1 <= u1 | U2 = u2) = dC/du2 (VineCopula's hfunc2)."""
    if family == "indep":
        return _clip(np.asarray(u1, float))
    return _clip(copula_h(transpose(family), _pdict(family, p), u2, u1))


def pair_h1inv(family: str, p: Sequence[float], u1: np.ndarray, w: np.ndarray) -> np.ndarray:
    """The u2 with h1(u1, u2) = w."""
    if family == "indep":
        return _clip(np.asarray(w, float))
    return _clip(copula_hinv(family, _pdict(family, p), u1, w))


def pair_h2inv(family: str, p: Sequence[float], u2: np.ndarray, w: np.ndarray) -> np.ndarray:
    """The u1 with h2(u1, u2) = w."""
    if family == "indep":
        return _clip(np.asarray(w, float))
    return _clip(copula_hinv(transpose(family), _pdict(family, p), u2, w))


def pair_tau(family: str, p: Sequence[float]) -> float:
    return 0.0 if family == "indep" else copula_tau(family, _pdict(family, p))


def pair_tails(family: str, p: Sequence[float]) -> tuple[float, float]:
    if family == "indep":
        return 0.0, 0.0
    td = tail_dependence(family, _pdict(family, p))
    lo, up = td["lower"], td["upper"]
    return (float(lo[0, 1]), float(up[0, 1])) if isinstance(lo, np.ndarray) else (float(lo), float(up))


def pair_bounds(family: str, p: Sequence[float] | None = None) -> list[tuple[float, float]]:
    if family == "gaussian":
        return [(-0.9999, 0.9999)]
    if family == "t":
        return [(-0.9999, 0.9999), (2.001, 60.0)]
    if family == "frank":
        return [(1e-4, 50.0)] if p is None or p[0] > 0 else [(-50.0, -1e-4)]
    return ARCH[family].bounds(2)


@_quiet
def fit_pair(u1: np.ndarray, u2: np.ndarray, family: str) -> tuple[tuple[float, ...], float]:
    """A pair copula's parameters by maximum likelihood, and its log likelihood."""
    if family == "indep":
        return (), 0.0
    if family == "gaussian":
        r0 = float(np.corrcoef(sps.norm.ppf(u1), sps.norm.ppf(u2))[0, 1])
        found = optimize.minimize_scalar(lambda r: -float(np.sum(pair_logpdf("gaussian", (r,), u1, u2))),
                                         bounds=(max(-0.9999, r0 - 0.2), min(0.9999, r0 + 0.2)), method="bounded",
                                         options={"xatol": 1e-10})
        return (float(found.x),), float(-found.fun)
    if family == "t":  # the profile likelihood of nu, maximised over rho at each (the quantiles once per nu)
        from scipy.special import gammaln, stdtrit

        def profile(nu: float) -> tuple[float, float]:
            x, y = stdtrit(nu, u1), stdtrit(nu, u2)
            const = gammaln((nu + 2) / 2) + gammaln(nu / 2) - 2 * gammaln((nu + 1) / 2)
            marg = (nu + 1) / 2 * (np.log1p(x * x / nu) + np.log1p(y * y / nu))
            s2, sxy = x * x + y * y, x * y

            def negll(r: float) -> float:
                q = (s2 - 2 * r * sxy) / (nu * (1 - r * r))
                return -float(np.sum(const - 0.5 * np.log1p(-r * r) - (nu + 2) / 2 * np.log1p(q) + marg))

            found = optimize.minimize_scalar(negll, bounds=(-0.9999, 0.9999), method="bounded", options={"xatol": 1e-10})
            return float(found.x), float(found.fun)

        lo, hi = pair_bounds("t")[1]
        grid = (2.5, 3.0, 4.0, 6.0, 9.0, 14.0, 22.0, 35.0, 55.0)
        vals = [profile(v)[1] for v in grid]
        i = int(np.argmin(vals))
        a, b = (grid[i - 1] if i > 0 else lo), (grid[i + 1] if i < len(grid) - 1 else hi)
        found = optimize.minimize_scalar(lambda v: profile(v)[1], bounds=(a, b), method="bounded", options={"xatol": 1e-7})
        r, f = profile(float(found.x))
        return (r, float(found.x)), -f
    fit = fit_copula(np.column_stack([u1, u2]), family)
    return tuple(float(fit[k]) for k in ARCH[family].names), float(fit["loglike"])


def _pair_se(family: str, p: Sequence[float], u1: np.ndarray, u2: np.ndarray) -> tuple[float, ...]:
    if not p:
        return ()

    def negll(v: np.ndarray) -> float:
        return -float(np.sum(pair_logpdf(family, v, u1, u2)))

    names = [str(i) for i in range(len(p))]
    se = _se_from(negll, np.asarray(p, float), names)
    return tuple(se[n] for n in names)


def independence_pvalue(u1: np.ndarray, u2: np.ndarray) -> float:
    """Genest and Favre's (2007) test of independence by Kendall's tau: sqrt(9 n (n - 1) / (2 (2 n + 5))) |tau| is
    asymptotically standard normal."""
    n = len(u1)
    tau = sps.kendalltau(u1, u2)[0]
    return float(2 * sps.norm.sf(np.sqrt(9 * n * (n - 1) / (2 * (2 * n + 5))) * abs(tau)))


def select_pair(u1: np.ndarray, u2: np.ndarray, families: Sequence[str] = DEFAULT_PAIR_FAMILIES, criterion: str = "aic",
                indeptest: bool = False, level: float = 0.05) -> tuple[str, tuple[float, ...], float]:
    """The pair copula of lowest AIC (or BIC) among ``families`` that fit the sign of Kendall's tau (independence
    as well when listed, or when the independence test does not reject at ``level``)."""
    if indeptest and independence_pvalue(u1, u2) >= level:
        return "indep", (), 0.0
    tau = sps.kendalltau(u1, u2)[0]
    allowed = _POSITIVE if tau > 0 else _NEGATIVE if tau < 0 else PAIR_FAMILIES
    todo = [f for f in families if f == "indep" or f in allowed]
    if not todo:
        raise SpecificationError("none of the pair-copula families fits the sign of an edge's Kendall's tau: add gaussian, t or frank")
    n = len(u1)
    best = None
    for f in todo:
        p, ll = fit_pair(u1, u2, f)
        k = n_pair_parameters(f)
        crit = (2 * k if criterion == "aic" else k * np.log(n)) - 2 * ll
        if best is None or crit < best[0]:
            best = (crit, f, p, ll)
    return best[1], best[2], best[3]


# -- the vine ----------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Edge:
    """An edge of a vine: the copula of F(a | cond) (its first argument) and F(b | cond) (its second)."""

    tree: int
    a: int
    b: int
    cond: tuple[int, ...]
    family: str = "indep"
    params: tuple[float, ...] = ()
    se: tuple[float, ...] = ()

    @property
    def pair(self) -> frozenset[int]:
        return frozenset((self.a, self.b))

    @property
    def full(self) -> frozenset[int]:
        return frozenset((self.a, self.b, *self.cond))

    def same_as(self, other: Edge) -> bool:
        return self.pair == other.pair and set(self.cond) == set(other.cond)

    def oriented(self, first: int) -> Edge:
        """The same edge with ``first`` as the copula's first argument (the family transposed if they swap)."""
        if first == self.a:
            return self
        return replace(self, a=self.b, b=self.a, family=transpose(self.family))


def _key(v: int, cond: Any) -> tuple[int, frozenset[int]]:
    return v, frozenset(cond)


def propagate(u: np.ndarray, trees: Sequence[Sequence[Edge]], need_last: bool = False) -> tuple[np.ndarray, dict]:
    """The log density at the rows of ``u`` (one value a row) and the conditional pseudo-observations F(v | D) of
    every tree, keyed (v, D)."""
    u = np.asarray(u, float)
    V: dict = {_key(j, ()): u[:, j] for j in range(u.shape[1])}
    ll = np.zeros(len(u))
    for t, tree in enumerate(trees):
        last = t == len(trees) - 1 and not need_last
        for e in tree:
            x, y = V[_key(e.a, e.cond)], V[_key(e.b, e.cond)]
            if e.family != "indep":
                ll = ll + pair_logpdf(e.family, e.params, x, y)
            if not last:
                V[_key(e.a, (*e.cond, e.b))] = pair_h2(e.family, e.params, x, y)
                V[_key(e.b, (*e.cond, e.a))] = pair_h1(e.family, e.params, x, y)
    return ll, V


def to_matrix(trees: Sequence[Sequence[Edge]], d: int) -> np.ndarray:
    """The vine's lower-triangular R-vine matrix (0-based variables): column i's edges join M[i, i] with M[k, i]
    given M[k+1:, i], tree d - 1 - k (Dissmann et al. 2013, section 2.2)."""
    left = [list(tree) for tree in trees]
    M = np.zeros((d, d), int)
    done: set[int] = set()
    for col in range(d - 1):
        top = d - 2 - col
        x = left[top][0].a
        M[col, col] = x
        for t in range(top, -1, -1):
            hits = [e for e in left[t] if x in e.pair]
            if len(hits) != 1:
                raise SpecificationError("the trees are not a regular vine")
            e = hits[0]
            M[d - 1 - t, col] = e.b if e.a == x else e.a
            left[t].remove(e)
        done.add(x)
    rest = set(range(d)) - done
    M[d - 1, d - 1] = rest.pop()
    for col in range(d - 1):  # the conditioning sets are those below in the column
        for k in range(col + 1, d):
            edge = _find(trees, M[col, col], M[k, col], M[k + 1:, col])
            if edge is None:
                raise SpecificationError("the trees are not a regular vine")
    return M


def _find(trees: Sequence[Sequence[Edge]], x: int, y: int, cond: Any) -> Edge | None:
    pair, cs = frozenset((int(x), int(y))), frozenset(int(c) for c in cond)
    t = len(cs)
    if t >= len(trees):
        return None
    for e in trees[t]:
        if e.pair == pair and frozenset(e.cond) == cs:
            return e
    return None


def from_matrix(M: Any, families: Any = None, par: Any = None, par2: Any = None, one_based: bool = True) -> list[list[Edge]]:
    """The trees of an R-vine matrix in VineCopula's convention (lower triangular; with ``one_based`` the variables
    numbered from 1) and, optionally, its family codes and parameters, the copula of each edge that of
    (F(M[k, i] | D), F(M[i, i] | D)) as in VineCopula."""
    M = np.asarray(M, int) - (1 if one_based else 0)
    d = len(M)
    trees: list[list[Edge]] = [[] for _ in range(d - 1)]
    for col in range(d - 1):
        for k in range(col + 1, d):
            t = d - 1 - k
            fam, p = "indep", ()
            if families is not None:
                code = int(np.asarray(families)[k, col])
                if code not in VC_FAMILY:
                    raise SpecificationError(f"VineCopula family {code} is not among the pair copulas here")
                fam = VC_FAMILY[code]
                raw = [float(np.asarray(par)[k, col]), float(np.asarray(par2)[k, col]) if par2 is not None else 0.0]
                p = tuple(abs(v) if fam in ROTATED and not fam.endswith("180") else v for v in raw[:n_pair_parameters(fam)])
            trees[t].append(Edge(t, int(M[k, col]), int(M[col, col]), tuple(sorted(int(c) for c in M[k + 1:, col])), fam, p))
    return trees


# -- structure selection -----------------------------------------------------------------------------------------------

def _max_spanning_tree(nodes: int, weights: dict[tuple[int, int], float]) -> list[tuple[int, int]]:
    """Prim's algorithm on the candidate edges (i < j) and their weights, the largest first."""
    inside, chosen = {0}, []
    while len(inside) < nodes:
        best = None
        for (i, j), w in weights.items():
            if (i in inside) != (j in inside) and (best is None or w > best[0]):
                best = (w, i, j)
        if best is None:
            raise SpecificationError("the candidate edges do not connect the tree")
        chosen.append((best[1], best[2]))
        inside |= {best[1], best[2]}
    return chosen


def _any_tree(nodes: int, candidates: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """A spanning tree by depth-first search (for the trees above a truncation level, VineCopula's choice)."""
    adj: dict[int, list[int]] = {i: [] for i in range(nodes)}
    for i, j in candidates:
        adj[i].append(j)
        adj[j].append(i)
    seen, out, stack = {0}, [], [0]
    while stack:
        i = stack.pop()
        for j in sorted(adj[i], reverse=True):
            if j not in seen:
                seen.add(j)
                out.append((min(i, j), max(i, j)))
                stack.append(j)
    return out


def best_path(W: np.ndarray) -> list[int]:
    """The order of the variables maximising the sum of W along the path: exact (Held-Karp) up to 12 variables,
    otherwise nearest neighbour from every start."""
    d = len(W)
    if d <= 2:
        return list(range(d))
    if d <= 12:
        full = (1 << d) - 1
        dp = np.full((1 << d, d), -np.inf)
        back = np.full((1 << d, d), -1, int)
        for j in range(d):
            dp[1 << j, j] = 0.0
        for mask in range(1, full + 1):
            for j in range(d):
                if not mask >> j & 1 or dp[mask, j] == -np.inf:
                    continue
                for k in range(d):
                    if mask >> k & 1:
                        continue
                    v = dp[mask, j] + W[j, k]
                    if v > dp[mask | 1 << k, k] + 1e-15:
                        dp[mask | 1 << k, k], back[mask | 1 << k, k] = v, j
        j, mask, path = int(np.argmax(dp[full])), full, []
        while j >= 0:
            path.append(j)
            j, mask = back[mask, j], mask & ~(1 << j)
        path = path[::-1]
        return path if path[0] < path[-1] else path[::-1]
    best = None
    for s in range(d):
        path, left = [s], set(range(d)) - {s}
        while left:
            k = max(left, key=lambda c: W[path[-1], c])
            path.append(k)
            left.remove(k)
        total = sum(W[a, b] for a, b in zip(path, path[1:]))
        if best is None or total > best[0]:
            best = (total, path)
    return best[1]


def _abs_tau(x: np.ndarray, y: np.ndarray) -> float:
    return abs(float(sps.kendalltau(x, y)[0]))


def select_structure(u: np.ndarray, structure: str = "rvine", order: Sequence[int] | None = None,
                     families: Sequence[str] = DEFAULT_PAIR_FAMILIES, criterion: str = "aic", indeptest: bool = False,
                     level: float = 0.05, trunc_level: int | None = None) -> list[list[Edge]]:
    """Dissmann et al.'s (2013) sequential selection: tree by tree, the maximum spanning tree of |Kendall's tau|
    (a star for a C-vine, the path of ``order`` or of the largest sum of |tau| for a D-vine), each edge's family by
    AIC or BIC and its parameters by maximum likelihood, then the next tree's pseudo-observations."""
    n, d = u.shape
    if structure not in STRUCTURES:
        raise SpecificationError(f"structure is one of {', '.join(STRUCTURES)}")
    trunc = d - 1 if trunc_level is None else int(trunc_level)
    if trunc < 1:
        raise SpecificationError("the truncation level is at least 1 (the first tree)")
    V: dict = {_key(j, ()): u[:, j] for j in range(d)}
    trees: list[list[Edge]] = []
    prev: list[Edge] = []
    for t in range(d - 1):
        if t == 0:
            nodes = [frozenset({j}) for j in range(d)]
            cand = {(i, j): (i, j, ()) for i in range(d) for j in range(i + 1, d)}
        else:
            nodes = [e.full for e in prev]
            cand = {}
            for i in range(len(prev)):
                for j in range(i + 1, len(prev)):
                    common = nodes[i] & nodes[j]
                    if len(common) == t:
                        (x,), (y,) = nodes[i] - common, nodes[j] - common
                        cand[(i, j)] = (x, y, tuple(sorted(common)))
        fitted_tree = t < trunc
        if fitted_tree:
            w = {k: _abs_tau(V[_key(x, c)], V[_key(y, c)]) for k, (x, y, c) in cand.items()}
        if structure == "dvine" and t == 0:
            if order is None:
                W = np.zeros((d, d))
                for (i, j), val in w.items():
                    W[i, j] = W[j, i] = val
                order = best_path(W)
            path = list(order)
            chosen = [(min(a, b), max(a, b)) for a, b in zip(path, path[1:])]
        elif structure == "cvine":
            if t == 0 and order is not None:
                root = int(order[0])
            elif t > 0 and order is not None:
                root = _cvine_root(prev, order, t)
            elif fitted_tree:
                sums = np.zeros(len(nodes))
                for (i, j), val in w.items():
                    sums[i] += val
                    sums[j] += val
                root = int(np.argmax(sums))
            else:
                root = 0
            chosen = [(min(root, j), max(root, j)) for j in range(len(nodes)) if j != root]
        elif fitted_tree:
            chosen = _max_spanning_tree(len(nodes), w)
        else:
            chosen = _any_tree(len(nodes), list(cand))
        tree = []
        for i, j in chosen:
            key = (min(i, j), max(i, j))
            if key not in cand:
                raise SpecificationError("the given order does not make a vine")
            x, y, c = cand[key]
            if fitted_tree:
                ux, uy = V[_key(x, c)], V[_key(y, c)]
                fam, p, _ = select_pair(ux, uy, families, criterion, indeptest, level)
                e = Edge(t, x, y, c, fam, p, _pair_se(fam, p, ux, uy))
                if t < d - 2:
                    V[_key(x, (*c, y))] = pair_h2(fam, p, ux, uy)
                    V[_key(y, (*c, x))] = pair_h1(fam, p, ux, uy)
            else:
                e = Edge(t, x, y, c)
            tree.append(e)
        trees.append(tree)
        prev = tree
    return trees


def _cvine_root(prev: list[Edge], order: Sequence[int], t: int) -> int:
    """In tree t of a C-vine with roots ``order``, the node (an edge of tree t - 1) joining the root of t - 1 and
    the next root."""
    for i, e in enumerate(prev):
        if e.pair == frozenset((order[t - 1], order[t])):
            return i
    raise SpecificationError("the given order does not make a C-vine")


def fixed_structure(d: int, structure: str, order: Sequence[int]) -> list[list[Edge]]:
    """The trees of a C-vine (``order`` its roots) or a D-vine (``order`` its path), all edges independence."""
    order = list(order)
    trees = []
    for t in range(d - 1):
        if structure == "dvine":
            trees.append([Edge(t, order[i], order[i + t + 1], tuple(sorted(order[i + 1:i + t + 1]))) for i in range(d - t - 1)])
        else:
            trees.append([Edge(t, order[t], order[j], tuple(sorted(order[:t]))) for j in range(t + 1, d)])
    return trees


def fit_parameters(u: np.ndarray, trees: Sequence[Sequence[Edge]], families: Sequence[str] = DEFAULT_PAIR_FAMILIES,
                   criterion: str = "aic", indeptest: bool = False, level: float = 0.05, select: bool = True,
                   trunc_level: int | None = None) -> list[list[Edge]]:
    """Each edge's family (``select``, by AIC or BIC) and parameters, sequentially, on a given structure."""
    V: dict = {_key(j, ()): u[:, j] for j in range(u.shape[1])}
    out = []
    trunc = len(trees) if trunc_level is None else int(trunc_level)
    for t, tree in enumerate(trees):
        new = []
        for e in tree:
            if t >= trunc:
                new.append(replace(e, family="indep", params=(), se=()))
                continue
            ux, uy = V[_key(e.a, e.cond)], V[_key(e.b, e.cond)]
            if select:
                fam, p, _ = select_pair(ux, uy, families, criterion, indeptest, level)
            else:
                fam = e.family
                p, _ = fit_pair(ux, uy, fam)
            new.append(replace(e, family=fam, params=p, se=_pair_se(fam, p, ux, uy)))
            V[_key(e.a, (*e.cond, e.b))] = pair_h2(fam, p, ux, uy)
            V[_key(e.b, (*e.cond, e.a))] = pair_h1(fam, p, ux, uy)
        out.append(new)
    return out


@_quiet
def refine_joint(u: np.ndarray, trees: Sequence[Sequence[Edge]]) -> list[list[Edge]]:
    """All the pair copulas' parameters by maximum likelihood together, from the sequential values."""
    slots = [(t, i) for t, tree in enumerate(trees) for i, e in enumerate(tree) if e.family != "indep"]
    x0, bounds, sizes = [], [], []
    for t, i in slots:
        e = trees[t][i]
        x0 += list(e.params)
        bounds += pair_bounds(e.family, e.params)
        sizes.append(len(e.params))
    x0 = np.clip(np.array(x0, float), [b[0] for b in bounds], [b[1] for b in bounds])

    def build(x: np.ndarray) -> list[list[Edge]]:
        new = [list(tree) for tree in trees]
        pos = 0
        for (t, i), k in zip(slots, sizes):
            new[t][i] = replace(new[t][i], params=tuple(float(v) for v in x[pos:pos + k]))
            pos += k
        return new

    def negll(x: np.ndarray) -> float:
        out = -float(np.sum(propagate(u, build(x))[0]))
        return out if np.isfinite(out) else 1e300

    if not slots:
        return [list(tree) for tree in trees]
    found = optimize.minimize(negll, x0, method="L-BFGS-B", bounds=bounds)
    x = found.x if found.fun <= negll(x0) else x0
    return build(x)


@_quiet
def simulate_vine(trees: Sequence[Sequence[Edge]], d: int, n: int, rng: np.random.Generator, w: np.ndarray | None = None) -> np.ndarray:
    """``n`` draws of the vine copula by the inverse Rosenblatt transform (``w``: the independent uniforms, one
    column a variable in the matrix's order from its last column)."""
    M = to_matrix(trees, d)
    w = rng.uniform(size=(n, d)) if w is None else np.asarray(w, float)
    V: dict = {}
    out = np.empty((n, d))
    last = int(M[d - 1, d - 1])
    V[_key(last, ())] = w[:, 0]
    out[:, last] = w[:, 0]
    for step, col in enumerate(range(d - 2, -1, -1), start=1):
        x = int(M[col, col])
        cur = w[:, step]
        rows = list(range(col + 1, d))  # from the highest tree (row col + 1) down to the first (row d - 1)
        for k in rows:
            y, cond = int(M[k, col]), [int(c) for c in M[k + 1:, col]]
            V[_key(x, (*cond, y))] = cur
            e = _find(trees, x, y, cond)
            uy = V[_key(y, cond)]
            cur = pair_h2inv(e.family, e.params, uy, cur) if e.a == x else pair_h1inv(e.family, e.params, uy, cur)
        V[_key(x, ())] = cur
        out[:, x] = cur
        for k in rows:  # the partners' conditional distributions given x, for the columns to the left
            y, cond = int(M[k, col]), [int(c) for c in M[k + 1:, col]]
            e = _find(trees, x, y, cond)
            ux, uy = V[_key(x, cond)], V[_key(y, cond)]
            V[_key(y, (*cond, x))] = pair_h1(e.family, e.params, ux, uy) if e.a == x else pair_h2(e.family, e.params, uy, ux)
    return out


# -- goodness of fit ---------------------------------------------------------------------------------------------------

def rosenblatt(u: np.ndarray, trees: Sequence[Sequence[Edge]], matrix: Any = None) -> np.ndarray:
    """The vine's Rosenblatt transform of the rows of ``u`` (the inverse of ``simulate_vine``): for each variable x
    of the R-vine matrix's diagonal, column i, its distribution given the variables below it in the column,
    F(x | M[i+1:, i]), from the h-functions (the last variable as it is). Independent uniforms if the vine is right.
    The order of conditioning is the matrix's: ``to_matrix(trees)`` or a given R-vine ``matrix`` of the same vine
    (VineCopula's, variables from 1), as RVinePIT takes it."""
    u = np.asarray(u, float)
    d = u.shape[1]
    M = to_matrix(trees, d) if matrix is None else np.asarray(matrix, int) - 1
    _, V = propagate(u, trees, need_last=True)
    Z = np.empty_like(u)
    for col in range(d):
        x = int(M[col, col])
        Z[:, x] = V[_key(x, [int(c) for c in M[col + 1:, col]])]
    return Z


def vine_gof_statistic(u: np.ndarray, trees: Sequence[Sequence[Edge]], matrix: Any = None) -> float:
    """VineCopula's ECP2 Cramer-von Mises statistic (RVineGofTest(..., method="ECP2")): the Rosenblatt transform Z
    of the data compared with independence through the empirical copula process, sum_t (C_n(Z_t) - prod_j Z_tj)^2,
    C_n(z) = #{s: Z_s <= z} / (T + 1); the Rosenblatt transform in the order of ``matrix`` (as ``rosenblatt``)."""
    Z = rosenblatt(u, trees, matrix)
    n = len(Z)
    from .copulas import empirical_copula

    return float(np.sum((empirical_copula(Z) * n / (n + 1) - np.prod(Z, axis=1)) ** 2))


def refit(u: np.ndarray, trees: Sequence[Sequence[Edge]], method: str = "sequential") -> list[list[Edge]]:
    """The same structure and families with their parameters re-estimated on ``u`` (sequentially, or ``joint``)."""
    out = fit_parameters(u, trees, select=False)
    return refine_joint(u, out) if method == "joint" else out


@_quiet
def vine_gof_test(u: np.ndarray, trees: Sequence[Sequence[Edge]], n_boot: int = 100, seed: int | None = 0,
                  method: str = "sequential") -> dict[str, float]:
    """The ECP2 statistic of the vine ``trees`` fitted to pseudo-observations ``u``, with a parametric bootstrap
    p-value that refits the vine: each sample drawn from the fitted vine, made pseudo-observations, its parameters
    re-estimated on the same structure and families (as the vine was: ``sequential`` or ``joint``), and its statistic
    computed with its own fit; p = (#{S_b >= S} + 1/2) / (B + 1). (VineCopula's own ECP2 p-value resamples the data
    with the parameters fixed; the parametric bootstrap here accounts for their estimation.)"""
    if n_boot < 1:
        raise SpecificationError("the goodness-of-fit bootstrap needs at least one sample")
    n, d = u.shape
    stat = vine_gof_statistic(u, trees)
    rng = random_number_generator(seed)
    boot = np.empty(int(n_boot))
    for b in range(int(n_boot)):
        Ub = pseudo_observations(simulate_vine(trees, d, n, rng))
        boot[b] = vine_gof_statistic(Ub, refit(Ub, trees, method))
    return {"statistic": stat, "pvalue": float((np.sum(boot >= stat) + 0.5) / (n_boot + 1)), "n_boot": float(n_boot)}


# -- the fitted vine ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class FittedVine:
    """A vine copula of several returns, with their margins: its trees of pair copulas, likelihood and simulation."""

    series: tuple[str, ...]
    structure: str
    trees: tuple[tuple[Edge, ...], ...]
    u: np.ndarray = field(repr=False)
    residuals: np.ndarray = field(repr=False)
    scale_next: np.ndarray = field(repr=False)
    mean_next: np.ndarray = field(repr=False)
    garch: bool = False
    index: Any = None
    margins: str = "empirical"
    method: str = "sequential"
    criterion: str = "aic"
    trunc_level: int | None = None
    margin_fits: tuple[Margin, ...] = ()
    margin_loglike: float | None = None
    gof_result: dict[str, float] | None = None

    stage = "finance"

    @property
    def d(self) -> int:
        return len(self.series)

    @property
    def edges(self) -> list[Edge]:
        return [e for tree in self.trees for e in tree]

    def logpdf(self, u: Any) -> np.ndarray:
        """The vine copula's log density at the rows of ``u``."""
        return propagate(np.atleast_2d(np.asarray(u, float)), self.trees)[0]

    def pdf(self, u: Any) -> np.ndarray:
        return np.exp(self.logpdf(u))

    @property
    def loglike(self) -> float:
        return float(np.sum(self.logpdf(self.u)))

    @property
    def k(self) -> int:
        return sum(n_pair_parameters(e.family) for e in self.edges)

    @property
    def aic(self) -> float:
        return 2 * self.k - 2 * self.loglike

    @property
    def bic(self) -> float:
        return self.k * np.log(len(self.u)) - 2 * self.loglike

    @property
    def total_loglike(self) -> float | None:
        return None if self.margin_loglike is None else self.loglike + self.margin_loglike

    def matrix(self) -> np.ndarray:
        """The R-vine matrix, variables numbered from 1 as in VineCopula."""
        return np.tril(to_matrix(self.trees, self.d) + 1)

    def to_vinecopula(self) -> dict[str, np.ndarray]:
        """The matrix, family codes and parameters (par, par2) in VineCopula's convention."""
        M0 = to_matrix(self.trees, self.d)
        fam, par, par2 = (np.zeros((self.d, self.d), int), np.zeros((self.d, self.d)), np.zeros((self.d, self.d)))
        for col in range(self.d - 1):
            for k in range(col + 1, self.d):
                e = _find(self.trees, M0[col, col], M0[k, col], M0[k + 1:, col]).oriented(int(M0[k, col]))
                fam[k, col] = VC_CODE[e.family]
                sign = -1 if e.family in ROTATED and not e.family.endswith("180") else 1
                if e.params:
                    par[k, col] = sign * e.params[0]
                if len(e.params) > 1:
                    par2[k, col] = e.params[1] if e.family == "t" or e.family.startswith("tawn") else sign * e.params[1]
        return {"matrix": np.tril(M0 + 1), "family": fam, "par": par, "par2": par2}

    def simulate(self, n: int = 10_000, seed: int | None = 0) -> np.ndarray:
        """``n`` draws of the copula's uniforms, one column a series."""
        return simulate_vine(self.trees, self.d, int(n), random_number_generator(seed))

    def rosenblatt(self, u: Any = None) -> np.ndarray:
        """The vine's Rosenblatt transform of ``u`` (by default the uniforms it was fitted to): independent uniforms
        if the vine is right."""
        return rosenblatt(self.u if u is None else np.atleast_2d(np.asarray(u, float)), self.trees)

    def goodness_of_fit(self, n_boot: int = 100, seed: int | None = 0) -> dict[str, float]:
        """The Rosenblatt-transform test of the vine (VineCopula's ECP2 Cramer-von Mises statistic) with a parametric
        bootstrap p-value that refits the vine's parameters on each sample (structure and families kept), on the
        pseudo-observations whatever the margins (refitted there first when the margins are parametric)."""
        U = pseudo_observations(self.residuals)
        trees = self.trees if self.margins == "empirical" else refit(U, self.trees, self.method)
        return vine_gof_test(U, trees, n_boot, seed, self.method)

    def tau_matrix(self, n: int = 20_000, seed: int | None = 0) -> np.ndarray:
        """Kendall's tau between every two series, from ``n`` draws (the first tree's pairs exactly)."""
        U = self.simulate(n, seed)
        T = np.eye(self.d)
        for i in range(self.d):
            for j in range(i + 1, self.d):
                T[i, j] = T[j, i] = sps.kendalltau(U[:, i], U[:, j])[0]
        for e in self.trees[0]:
            T[e.a, e.b] = T[e.b, e.a] = pair_tau(e.family, e.params)
        return T

    def portfolio_var(self, weights: Sequence[float], level: float = 0.99, n: int = 100_000, margins: str | None = None,
                      seed: int | None = 0) -> dict[str, float]:
        """Next period's VaR and ES of a portfolio by simulation from the vine and the margins (as for ``copula``)."""
        from .copulas import _portfolio_from_uniforms

        return _portfolio_from_uniforms(self, self.simulate(n, seed), weights, level, margins)

    def summary(self) -> Table:
        """The structure: tree, edge, family, parameters, Kendall's tau and tail dependence of each pair copula."""
        rows, names = [], []
        s = self.series
        for e in self.edges:
            p = list(e.params) + [np.nan] * (2 - len(e.params))
            se = list(e.se) + [np.nan] * (2 - len(e.se))
            lo, up = pair_tails(e.family, e.params)
            cond = ("; " + ", ".join(s[c] for c in e.cond)) if e.cond else ""
            names.append(f"{s[e.a]}, {s[e.b]}{cond}")
            rows.append([e.tree + 1, e.family, p[0], se[0], p[1], se[1], pair_tau(e.family, e.params), lo, up])
        T = len(self.u)
        what = "GARCH(1,1) standardised residuals" if self.garch else "the returns"
        how = {"empirical": "their ranks", "normal": "fitted normal margins", "t": "fitted t margins",
               "skewt": "fitted skewed-t margins (Hansen)"}[self.margins]
        kind = {"rvine": "R-vine", "cvine": "C-vine", "dvine": "D-vine"}[self.structure]
        notes = [f"{kind} copula of {what} by {how}, {T} periods, {self.d} series; families by {self.criterion.upper()}, "
                 + ("parameters jointly by maximum likelihood from the sequential estimates." if self.method == "joint"
                    else "parameters sequentially (tree by tree) by maximum likelihood."),
                 f"Log likelihood: {self.loglike:.4f}", f"Parameters: {self.k}", f"Akaike info criterion: {self.aic:.4f}",
                 f"Schwarz criterion: {self.bic:.4f}",
                 *([f"Log likelihood (with the margins): {self.total_loglike:.4f}"] if self.total_loglike is not None else []),
                 "An edge 'a, b; D' is the copula of a and b given D (its first argument a). Standard errors from each "
                 "pair's observed information with its inputs taken as known; parameter 2 is the t's degrees of "
                 "freedom, the BB family's delta or Tawn's psi."]
        if self.trunc_level is not None:
            notes.append(f"Truncated after tree {self.trunc_level}: the pair copulas above are independence.")
        if self.gof_result is not None:
            g = self.gof_result
            notes.append(f"Goodness of fit (Rosenblatt transform, Cramer-von Mises against independence, VineCopula's ECP2): "
                         f"statistic {g['statistic']:.6g}, p-value {g['pvalue']:.4f} (parametric bootstrap, "
                         f"{int(g['n_boot'])} samples, the parameters refitted on each).")
        return Table(["tree", "family", "parameter", "std err", "parameter 2", "std err 2", "Kendall's tau", "lower tail",
                      "upper tail"], rows, index=names, index_name="edge", title="Vine copula", notes=tuple(notes))

    def plot(self) -> Figure:
        from pymetrix.figures import Figure, Panel, Scatter

        e = self.trees[0][0]
        sim = self.simulate(len(self.u), seed=1)
        a, b = self.series[e.a], self.series[e.b]
        return Figure((Panel((Scatter(pd.Series(self.u[:, e.a]), pd.Series(self.u[:, e.b]), "data"),), title=f"Pseudo-observations: {a} and {b}",
                             xlabel=a, ylabel=b),
                       Panel((Scatter(pd.Series(sim[:, e.a]), pd.Series(sim[:, e.b]), "simulated"),), title="From the fitted vine",
                             xlabel=a, ylabel=b)))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from pymetrix.extension import views_frame

            return views_frame(self, view)
        return pd.DataFrame(self.u, columns=list(self.series), index=self.index)

    def __repr__(self) -> str:
        return f"<FittedVine {self.structure} of {', '.join(self.series)}>"


def pair_family_list(families: Any) -> list[str]:
    """The pair families to select among: ``None`` the default set (``DEFAULT_PAIR_FAMILIES``), ``"all"`` every one
    (BB6, BB8, Tawn and the rotated two-parameter families too), or the names, as a list or separated by commas."""
    if families is None:
        return list(DEFAULT_PAIR_FAMILIES)
    if isinstance(families, str):
        if families.strip().lower() == "all":
            return list(PAIR_FAMILIES)
        return [f.strip() for f in families.split(",") if f.strip()]
    out: list[str] = []
    for f in families:
        out += list(PAIR_FAMILIES) if f == "all" else [f]
    return out


def fit_vine(u: np.ndarray, structure: str = "rvine", order: Sequence[int] | None = None, matrix: Any = None,
             families: Sequence[str] | None = None, criterion: str = "aic", indeptest: bool = False, level: float = 0.05,
             trunc_level: int | None = None, method: str = "sequential") -> list[list[Edge]]:
    """A vine copula's trees on uniforms ``u``: the structure selected (or given by ``order`` for a C- or D-vine, or
    by an R-vine ``matrix``, 1-based), each pair's family by ``criterion`` and its parameters, sequentially or
    ``joint``."""
    u = np.asarray(u, float)
    d = u.shape[1]
    families = pair_family_list(families)
    bad = [f for f in families if f not in PAIR_FAMILIES]
    if bad:
        raise SpecificationError(f"unknown pair-copula families {', '.join(map(repr, bad))}: they are {', '.join(PAIR_FAMILIES)}")
    if criterion not in ("aic", "bic"):
        raise SpecificationError("criterion is 'aic' or 'bic'")
    if method not in ("sequential", "joint"):
        raise SpecificationError("method is 'sequential' or 'joint'")
    if trunc_level is not None and not 1 <= int(trunc_level) <= d - 1:
        raise SpecificationError(f"the truncation level is between 1 and {d - 1} (the number of trees)")
    if matrix is not None:
        M = np.asarray(matrix, int)
        if M.shape != (d, d) or sorted(np.diag(M)) != list(range(1, d + 1)):
            raise SpecificationError(f"the R-vine matrix is {d} by {d}, its diagonal the series numbered from 1")
        trees = fit_parameters(u, from_matrix(M), families, criterion, indeptest, level, trunc_level=trunc_level)
    else:
        trees = select_structure(u, structure, order, families, criterion, indeptest, level, trunc_level)
    if method == "joint":
        trees = refine_joint(u, trees)
    return trees


with plugin_context(PLUGIN):
    @api("finance:vine_copula")
    def vine_copula(data: Dataset, series: Sequence[str], structure: Literal["rvine", "cvine", "dvine"] = "rvine",
                    order: Sequence[str] | None = None, families: Sequence[str] | None = None,
                    criterion: Literal["aic", "bic"] = "aic", indeptest: bool = False, level: float = 0.05,
                    trunc_level: int | None = None, method: Literal["sequential", "joint"] = "sequential",
                    garch: bool = False, dist: Literal["normal", "t"] | None = None,
                    margins: Literal["empirical", "normal", "t", "skewt"] = "empirical",
                    matrix: Sequence[Sequence[int]] | None = None, gof: bool = False, n_boot: int = 100,
                    seed: int = 0) -> FittedVine:
        """A vine copula of the returns ``series``: an R-vine (or a C- or D-vine) whose trees are selected by
        Dissmann et al.'s maximum spanning trees of |Kendall's tau|, each pair copula's family by AIC or BIC among
        ``families`` (all the plugin's bivariate families, rotations and independence), estimated tree by tree or
        ``joint``; ``order`` fixes a D-vine's path or a C-vine's roots, ``matrix`` an R-vine matrix (VineCopula's,
        variables numbered from 1); ``trunc_level`` makes the trees above it independence; ``families="all"`` adds
        BB6, BB8, Tawn and the rotated two-parameter families to those selected among. Margins as for ``copula``;
        ``gof`` adds the Rosenblatt-transform goodness-of-fit test (``n_boot`` parametric bootstrap samples, each
        refitted)."""
        series = [series] if isinstance(series, str) else list(series)
        if margins not in MARGINS:
            raise SpecificationError(f"margins is one of {', '.join(MARGINS)}")
        if order is not None:
            order = [order] if isinstance(order, str) else list(order)
            if structure == "rvine":
                raise SpecificationError("order is for a C-vine (its roots) or a D-vine (its path)")
            if structure == "dvine" and sorted(order) != sorted(series):
                raise SpecificationError("a D-vine's order lists every series once")
            if structure == "cvine" and (len(set(order)) != len(order) or not set(order) <= set(series)):
                raise SpecificationError("a C-vine's order lists its roots, series of the copula, each once")
        prep = _prepare(data, series, garch, dist, margins)
        idx = None
        if order is not None:
            idx = [series.index(s) for s in order]
            if structure == "cvine":
                idx += [j for j in range(len(series)) if j not in idx]
        if matrix is not None:
            structure = "rvine"
        trees = fit_vine(prep.u, structure, idx, matrix, families, criterion, indeptest, level, trunc_level, method)
        fit = FittedVine(tuple(series), structure, tuple(tuple(t) for t in trees), prep.u, prep.resid, prep.scale, prep.mean,
                         bool(garch), prep.frame.index, margins, method, criterion, trunc_level, prep.margin_fits,
                         prep.margin_loglike)
        if gof:
            fit = replace(fit, gof_result=fit.goodness_of_fit(int(n_boot), seed))
        return fit


__all__ = ["FittedVine", "vine_copula", "fit_vine", "pseudo_observations", "rosenblatt", "vine_gof_test"]
