"""Functions of several arguments: probability distributions and moving-window statistics.

Distribution functions are named as in MATLAB and Dynare: ``<family>pdf`` (the density, or for Poisson and binomial
the probability mass), ``<family>cdf`` and ``<family>inv`` (the inverse of the cdf, the quantile function), for the
families below. Arguments after the first are the distribution's parameters, in MATLAB's order and
parameterisation; those with defaults may be left out (``normcdf(x)`` is the standard normal)::

    norm   mu = 0, sigma = 1       normal with mean mu and standard deviation sigma
    t      df                      Student's t
    chi2   df                      chi-squared
    f      d1, d2                  F with d1 numerator and d2 denominator degrees of freedom
    gam    a, b                    gamma with shape a and scale b (mean a*b, as MATLAB's gampdf)
    beta   a, b                    beta
    logn   mu = 0, sigma = 1       lognormal: log x is normal(mu, sigma)
    unif   a = 0, b = 1            uniform on [a, b]
    exp    mu = 1                  exponential with mean mu (MATLAB's parameterisation, not the rate)
    poiss  lambda                  Poisson
    bino   n, p                    binomial

``poissinv`` and ``binoinv`` are MATLAB's: the smallest integer k with cdf(k) >= p (so ``poissinv(0, lambda)`` is 0
and ``binoinv(1, n, p)`` is n, ``poissinv(1, lambda)`` infinite), as scipy's ``ppf``. They are steps in p, with no
derivative: linearising through one is refused with a message.

``erf`` and ``erfc`` are the error function and its complement. Invalid parameters (a negative standard deviation,
a probability outside [0, 1]) give missing values, as in MATLAB; missing values stay missing.

Moving-window statistics (``movavg(x, n)`` and the rest in ``MOVING``) are trailing windows of n periods over the
time axis: the value at t uses x[t], x[t-1], ..., x[t-n+1], and is missing until a full window is there (or when any
value in it is missing). The parser writes them with every lag explicit, ``Call("movavg", (x, x[-1], ..., x[-n+1]))``,
so lags work within each entity of a panel, respect gaps in the dates, and are seen by everything that reads an
expression's lags (models, Jacobians, compiled solvers); they print back as typed. Variances, covariances and
correlations use the n - 1 denominator (as pandas' rolling functions do).

Centred windows put the window around t: ``cmovavg(x, 5)`` uses x[+2], ..., x[-2] (an even n one more lag than
leads: ``cmovavg(x, 4)`` is x[+1] ... x[-2], as pandas' ``rolling(n, center=True)``), stored with its leads explicit,
so a model reads them as leads (and is solved with a forward-looking solver); missing at both ends until the window
is full. Every moving function has its centred one (``cmovsum``, ``cmovsd``, ..., ``cmovbeta``), with the same
derivatives.

Expanding windows use every period up to t, within each entity of a panel: ``cumavg(x)``, ``cumsum``, ``cumsd``,
``cumvar``, ``cummin``, ``cummax``, ``cumcov(x, y)``, ``cumcorr(x, y)`` and ``cumbeta(y, x)``, as pandas'
``expanding()`` (missing values skipped; a value from the first observation, the second for variances). They cannot
be written with fixed lags, so, as the group-wise functions, they work on series (``data.eval``, derived series,
linear-form regressors) and are refused in models, with no derivative.

Derivatives: ``partials`` gives each argument's partial derivative as an expression, or None where none is offered
(the degrees of freedom of t, chi-squared and F, the shape parameters of gamma and beta, the binomial's n, the
moving minimum and maximum, the Poisson and binomial quantiles and the expanding windows); differentiating through one of those is refused with a message rather than
approximated. The discrete distributions are flat in k between integers, as ``step()`` is flat away from 0.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .nodes import CENTRED, EXPANDING, MOVING, Call, Expr, Num, add, div, mul, neg, power, sub

__all__ = ["CENTRED", "DISTRIBUTIONS", "EXPANDING", "FUNCTIONS", "MOVING", "SPECIAL", "arity", "expanding", "numeric", "partials",
           "signature", "window"]


@dataclass(frozen=True)
class Family:
    params: tuple[str, ...]
    defaults: tuple[float | None, ...]  # None: required
    kinds: tuple[str, ...] = ("pdf", "cdf", "inv")


FAMILIES: dict[str, Family] = {
    "norm": Family(("mu", "sigma"), (0.0, 1.0)),
    "t": Family(("df",), (None,)),
    "chi2": Family(("df",), (None,)),
    "f": Family(("d1", "d2"), (None, None)),
    "gam": Family(("a", "b"), (None, None)),
    "beta": Family(("a", "b"), (None, None)),
    "logn": Family(("mu", "sigma"), (0.0, 1.0)),
    "unif": Family(("a", "b"), (0.0, 1.0)),
    "exp": Family(("mu",), (1.0,)),
    "poiss": Family(("lambda",), (None,)),
    "bino": Family(("n", "p"), (None, None)),
}

# name -> (family, kind)
DISTRIBUTIONS: dict[str, tuple[str, str]] = {f"{fam}{kind}": (fam, kind) for fam, spec in FAMILIES.items() for kind in spec.kinds}
SPECIAL = frozenset({"erf", "erfc"})
FUNCTIONS = frozenset(DISTRIBUTIONS) | SPECIAL | frozenset(MOVING) | frozenset(EXPANDING)


def arity(fn: str) -> tuple[int, int]:
    """The fewest and most arguments ``fn`` takes (for a distribution function)."""
    if fn in SPECIAL:
        return 1, 1
    fam = FAMILIES[DISTRIBUTIONS[fn][0]]
    required = sum(d is None for d in fam.defaults)
    return 1 + required, 1 + len(fam.params)


def full_args(fn: str, args: Sequence[Any], number: Callable[[float], Any] = float) -> list[Any]:
    """``args`` with the defaults of the arguments left out filled in."""
    if fn in SPECIAL:
        return list(args)
    fam = FAMILIES[DISTRIBUTIONS[fn][0]]
    return list(args) + [number(d) for d in fam.defaults[len(args) - 1:]]  # type: ignore[arg-type]


def signature(fn: str) -> str:
    """``tcdf(x, df)``, ``normcdf(x, mu = 0, sigma = 1)``: for messages and help."""
    if fn in MOVING:
        return f"{fn}(x, y, n)" if MOVING[fn] == 2 else f"{fn}(x, n)"
    if fn in EXPANDING:
        return f"{fn}(x, y)" if MOVING[EXPANDING[fn]] == 2 else f"{fn}(x)"
    if fn in SPECIAL:
        return f"{fn}(x)"
    family, kind = DISTRIBUTIONS[fn]
    fam = FAMILIES[family]
    first = "p" if kind == "inv" else "k" if family in ("poiss", "bino") else "x"
    rest = [p if d is None else f"{p} = {d:g}" for p, d in zip(fam.params, fam.defaults)]
    return f"{fn}({', '.join([first, *rest])})"


# numbers


def _keep_series(value: Any, args: Sequence[Any]) -> Any:
    series = next((a for a in args if isinstance(a, pd.Series)), None)
    if series is not None and not isinstance(value, pd.Series):
        return pd.Series(np.asarray(value, dtype=float), index=series.index)
    return value


def _arrays(args: Sequence[Any]) -> list[Any]:
    return [a.to_numpy(float) if isinstance(a, pd.Series) else a for a in args]


def _normal(kind: str, x: Any, mu: Any, s: Any) -> Any:
    from scipy import special

    x, mu, s = (np.asarray(v, dtype=float) for v in (x, mu, s))
    bad = ~(s > 0)
    if kind == "inv":
        out = mu + s * special.ndtri(x)
    else:
        z = (x - mu) / np.where(bad, 1.0, s)
        out = special.ndtr(z) if kind == "cdf" else np.exp(-0.5 * z * z) / (np.sqrt(2 * np.pi) * np.where(bad, 1.0, s))
    out = np.where(bad, np.nan, out)
    return out[()] if out.ndim == 0 else out


def _scipy(family: str, kind: str, x: Any, params: Sequence[Any]) -> Any:
    from scipy import stats

    if family == "t":
        dist, a, kw = stats.t, (params[0],), {}
    elif family == "chi2":
        dist, a, kw = stats.chi2, (params[0],), {}
    elif family == "f":
        dist, a, kw = stats.f, (params[0], params[1]), {}
    elif family == "gam":
        dist, a, kw = stats.gamma, (params[0],), {"scale": params[1]}
    elif family == "beta":
        dist, a, kw = stats.beta, (params[0], params[1]), {}
    elif family == "logn":
        dist, a, kw = stats.lognorm, (params[1],), {"scale": np.exp(params[0])}
    elif family == "unif":
        dist, a, kw = stats.uniform, (), {"loc": params[0], "scale": np.subtract(params[1], params[0])}
    elif family == "exp":
        dist, a, kw = stats.expon, (), {"scale": params[0]}
    elif family == "poiss":
        dist, a, kw = stats.poisson, (params[0],), {}
    else:
        dist, a, kw = stats.binom, (params[0], params[1]), {}
    if kind == "inv":
        q = dist.ppf(x, *a, **kw)
        if family in ("poiss", "bino"):
            # The parameters and p are checked first (old SciPy does not give nan for every invalid parameter), then
            # MATLAB's edge rules: at least 0 (scipy's ppf(0) is -1), and 0 where all the mass is at 0.
            pv = np.asarray(x, dtype=float)
            if family == "poiss":
                lam = np.asarray(a[0], dtype=float)
                invalid = np.isnan(lam) | (lam < 0)
                certain = lam == 0
            else:
                n, pr = np.asarray(a[0], dtype=float), np.asarray(a[1], dtype=float)
                invalid = np.isnan(n) | (n < 0) | (n != np.floor(n)) | np.isnan(pr) | (pr < 0) | (pr > 1)
                certain = pr == 0
            invalid = invalid | np.isnan(pv) | (pv < 0) | (pv > 1)
            q = np.where(q == -1, 0.0, q)
            q = np.where(certain, 0.0, q)
            q = np.where(invalid, np.nan, q)
        return q
    if kind == "cdf":
        return dist.cdf(x, *a, **kw)
    return dist.pmf(x, *a, **kw) if family in ("poiss", "bino") else dist.pdf(x, *a, **kw)


def numeric(fn: str, args: Sequence[Any]) -> Any:
    """The value of distribution function ``fn`` (or erf, erfc) at ``args`` (numbers, arrays or Series)."""
    from scipy import special

    values = full_args(fn, _arrays(args))
    with np.errstate(all="ignore"):
        if fn == "erf":
            out = special.erf(values[0])
        elif fn == "erfc":
            out = special.erfc(values[0])
        else:
            family, kind = DISTRIBUTIONS[fn]
            if family == "norm":
                out = _normal(kind, *values)
            else:
                out = _scipy(family, kind, values[0], values[1:])
                if family in ("poiss", "bino"):  # missing in, missing out (scipy gives 0 for a missing k)
                    out = np.where(np.isnan(np.asarray(values[0], dtype=float)), np.nan, out)
                    out = out[()] if np.ndim(out) == 0 else out
    return _keep_series(out, args)


def window(fn: str, args: Sequence[Any]) -> Any:
    """A moving-window statistic from the window's values: ``args`` are x, x[-1], ..., (then y, y[-1], ...); a
    centred window's from its longest lead down."""
    fn = CENTRED.get(fn, fn)
    k = MOVING[fn]
    n = len(args) // k
    arrays = [np.asarray(a.to_numpy(float) if isinstance(a, pd.Series) else a, dtype=float) for a in args]
    stacked = np.stack(np.broadcast_arrays(*arrays))
    x = stacked[:n]
    with np.errstate(all="ignore"):
        if fn == "movavg":
            out = x.mean(axis=0)
        elif fn == "movsum":
            out = x.sum(axis=0)
        elif fn == "movmin":
            out = np.min(x, axis=0)
        elif fn == "movmax":
            out = np.max(x, axis=0)
        elif fn in ("movvar", "movsd"):
            out = x.var(axis=0, ddof=1)
            out = np.sqrt(out) if fn == "movsd" else out
        else:
            y = stacked[n:]
            dx, dy = x - x.mean(axis=0), y - y.mean(axis=0)
            sxy, sxx, syy = (dx * dy).sum(axis=0), (dx * dx).sum(axis=0), (dy * dy).sum(axis=0)
            if fn == "movcov":
                out = sxy / (n - 1)
            elif fn == "movcorr":
                out = sxy / np.sqrt(sxx * syy)
            else:  # movbeta(y, x, n): the slope of the first on the second
                out = sxy / syy
    out = out[()] if out.ndim == 0 else out
    return _keep_series(out, args)


# derivatives


def _call(fn: str, *args: Expr) -> Expr:
    """``fn(args)``, leaving out trailing arguments at their defaults (``normpdf(x)`` rather than ``normpdf(x, 0, 1)``)."""
    if fn in DISTRIBUTIONS:
        fam = FAMILIES[DISTRIBUTIONS[fn][0]]
        out = list(args)
        while len(out) > 1 and fam.defaults[len(out) - 2] is not None and out[-1] == Num(float(fam.defaults[len(out) - 2])):  # type: ignore[arg-type]
            out.pop()
        args = tuple(out)
    return Call(fn, tuple(args))


def _one(e: Expr) -> Expr:
    return div(Num(1.0), e)


def partials(node: Call) -> list[Expr | None]:
    """The partial derivative of ``node`` with respect to each of its arguments; None where none is offered."""
    fn = node.fn
    if fn in MOVING or fn in EXPANDING:
        return _moving_partials(node)
    if fn == "erf":
        return [mul(Num(2.0 / np.sqrt(np.pi)), Call("exp", (neg(power(node.args[0], Num(2.0))),)))]
    if fn == "erfc":
        return [mul(Num(-2.0 / np.sqrt(np.pi)), Call("exp", (neg(power(node.args[0], Num(2.0))),)))]
    family, kind = DISTRIBUTIONS[fn]
    x, *p = full_args(fn, node.args, Num)
    pdf = _call(f"{family}pdf", x, *p)
    cdf = _call(f"{family}cdf", x, *p)
    if kind == "inv" and family in ("poiss", "bino"):  # a step in p: no derivative offered
        return [None] * (1 + len(p))
    if kind == "inv":  # q = inv(p): dq/dp = 1 / pdf(q)
        q = node
        dp = _one(_call(f"{family}pdf", q, *p))
        if family == "norm":
            mu, s = p
            return [dp, Num(1.0), div(sub(q, mu), s)]
        if family == "logn":
            mu, s = p
            return [dp, q, div(mul(q, sub(Call("log", (q,)), mu)), s)]
        if family == "gam":
            return [dp, None, div(q, p[1])]
        if family == "unif":
            a, b = p
            return [sub(b, a), sub(Num(1.0), x), x]
        if family == "exp":
            return [dp, div(q, p[0])]
        return [dp] + [None] * len(p)
    if kind == "cdf":
        if family == "norm":
            mu, s = p
            return [pdf, neg(pdf), neg(mul(div(sub(x, mu), s), pdf))]
        if family == "logn":
            mu, s = p
            return [pdf, neg(mul(x, pdf)), neg(mul(div(mul(sub(Call("log", (x,)), mu), x), s), pdf))]
        if family == "gam":
            return [pdf, None, neg(mul(div(x, p[1]), pdf))]
        if family == "unif":
            return [pdf, mul(sub(cdf, Num(1.0)), pdf), neg(mul(cdf, pdf))]
        if family == "exp":
            return [pdf, neg(mul(div(x, p[0]), pdf))]
        if family == "poiss":  # flat in k; d/dlambda = -pmf(floor(k)) = -(cdf(k) - cdf(k - 1))
            lam = p[0]
            return [Num(0.0), neg(sub(node, _call("poisscdf", sub(x, Num(1.0)), lam)))]
        if family == "bino":  # d/dp = -n pmf(floor(k); n - 1, p)
            n, pr = p
            below = sub(_call("binocdf", x, sub(n, Num(1.0)), pr), _call("binocdf", sub(x, Num(1.0)), sub(n, Num(1.0)), pr))
            return [Num(0.0), None, neg(mul(n, below))]
        return [pdf] + [None] * len(p)
    # densities: d pdf = pdf * d log pdf
    f = node
    if family == "norm":
        mu, s = p
        z = div(sub(x, mu), power(s, Num(2.0)))
        return [neg(mul(z, f)), mul(z, f), mul(f, sub(div(power(sub(x, mu), Num(2.0)), power(s, Num(3.0))), _one(s)))]
    if family == "logn":
        mu, s = p
        z = sub(Call("log", (x,)), mu)
        return [neg(mul(div(f, x), add(Num(1.0), div(z, power(s, Num(2.0)))))), mul(f, div(z, power(s, Num(2.0)))),
                mul(f, sub(div(power(z, Num(2.0)), power(s, Num(3.0))), _one(s)))]
    if family == "t":
        v = p[0]
        return [neg(mul(div(mul(add(v, Num(1.0)), x), add(v, power(x, Num(2.0)))), f)), None]
    if family == "chi2":
        v = p[0]
        return [mul(sub(div(sub(div(v, Num(2.0)), Num(1.0)), x), Num(0.5)), f), None]
    if family == "f":
        d1, d2 = p
        first = div(sub(div(d1, Num(2.0)), Num(1.0)), x)
        second = mul(div(add(d1, d2), Num(2.0)), div(d1, add(d2, mul(d1, x))))
        return [mul(sub(first, second), f), None, None]
    if family == "gam":
        a, b = p
        return [mul(sub(div(sub(a, Num(1.0)), x), _one(b)), f), None, mul(f, sub(div(x, power(b, Num(2.0))), div(a, b)))]
    if family == "beta":
        a, b = p
        return [mul(sub(div(sub(a, Num(1.0)), x), div(sub(b, Num(1.0)), sub(Num(1.0), x))), f), None, None]
    if family == "unif":
        a, b = p
        return [Num(0.0), div(f, sub(b, a)), neg(div(f, sub(b, a)))]
    if family == "exp":
        mu = p[0]
        return [neg(div(f, mu)), mul(f, div(sub(div(x, mu), Num(1.0)), mu))]
    if family == "poiss":
        lam = p[0]
        return [Num(0.0), mul(f, sub(div(x, lam), Num(1.0)))]
    n, pr = p  # binomial
    return [Num(0.0), None, mul(f, sub(div(x, pr), div(sub(n, x), sub(Num(1.0), pr))))]


def expanding(fn: str, args: Sequence[Any], time: str | None = None) -> pd.Series:
    """An expanding-window statistic (``cumavg`` and the rest in ``EXPANDING``) of series ``args`` (x, or x and y),
    within each entity of a panel (every index level but the time axis ``time``; without it, the level of periods or
    dates, else the last), in time order, as pandas' ``expanding()``."""
    from .nodes import ExpressionError

    if not all(isinstance(a, pd.Series) for a in args):
        raise ExpressionError(f"{fn}() is an expanding window: it works on series (data.eval, derived series, linear-form "
                              "regressors), not inside estimation or model solvers; make it a derived series first")
    x = args[0].astype(float)
    y = args[1].astype(float).reindex(x.index) if len(args) > 1 else None
    index = x.index
    if isinstance(index, pd.MultiIndex):
        names = list(index.names)
        if time not in names:
            kinds = [i for i in range(index.nlevels)
                     if isinstance(index.levels[i], (pd.PeriodIndex, pd.DatetimeIndex))]
            time_level = kinds[-1] if kinds else index.nlevels - 1
        else:
            time_level = names.index(time)
        keys = [index.get_level_values(i) for i in range(index.nlevels) if i != time_level]
        order = np.lexsort([np.asarray(index.get_level_values(time_level))] + [np.asarray(k) for k in reversed(keys)])
        groups = pd.MultiIndex.from_arrays(keys).take(order) if len(keys) > 1 else keys[0].take(order)
    else:
        order = np.argsort(np.asarray(index), kind="stable") if index.is_monotonic_increasing is False else np.arange(len(index))
        groups = np.zeros(len(index))
    xs = pd.Series(x.to_numpy()[order])
    ys = pd.Series(y.to_numpy()[order]) if y is not None else None
    out = np.full(len(xs), np.nan)
    codes = pd.factorize(groups)[0] if not isinstance(groups, np.ndarray) else groups.astype(int)
    base = EXPANDING[fn]
    for code in np.unique(codes):
        rows = np.flatnonzero(codes == code)
        a = xs.iloc[rows].reset_index(drop=True)
        if base in ("movavg", "movsum", "movmin", "movmax", "movvar", "movsd"):
            stat = {"movavg": "mean", "movsum": "sum", "movmin": "min", "movmax": "max", "movvar": "var", "movsd": "std"}[base]
            values = getattr(a.expanding(), stat)()
        else:
            b = ys.iloc[rows].reset_index(drop=True)  # type: ignore[union-attr]
            both = a.notna() & b.notna()
            a, b = a.where(both), b.where(both)
            if base == "movcov":
                values = a.expanding().cov(b)
            elif base == "movcorr":
                values = a.expanding().corr(b)
            else:  # cumbeta(y, x): the slope of the first on the second
                values = a.expanding().cov(b) / b.expanding().var()
        out[rows] = values.to_numpy(float)
    result = np.empty(len(out))
    result[order] = out
    return pd.Series(result, index=index)


def _moving_partials(node: Call) -> list[Expr | None]:
    if node.fn in EXPANDING:
        return [None] * len(node.args)
    fn, args = CENTRED.get(node.fn, node.fn), node.args
    k = MOVING[fn]
    n = len(args) // k
    xs = args[:n]
    if fn == "movavg":
        return [Num(1.0 / n)] * n
    if fn == "movsum":
        return [Num(1.0)] * n
    if fn in ("movmin", "movmax"):
        return [None] * n
    mean_x = Call("movavg", xs)
    dev_x = [sub(a, mean_x) for a in xs]
    if fn == "movvar":
        return [mul(Num(2.0 / (n - 1)), d) for d in dev_x]
    if fn == "movsd":
        return [div(d, mul(Num(n - 1.0), node)) for d in dev_x]
    ys = args[n:]
    mean_y = Call("movavg", ys)
    dev_y = [sub(a, mean_y) for a in ys]
    if fn == "movcov":
        return [mul(Num(1.0 / (n - 1)), d) for d in dev_y] + [mul(Num(1.0 / (n - 1)), d) for d in dev_x]
    sxx = mul(Num(n - 1.0), Call("movvar", xs))
    syy = mul(Num(n - 1.0), Call("movvar", ys))
    if fn == "movbeta":  # b = Sxy / Syy (the first series on the second)
        return [div(d, syy) for d in dev_y] + [div(sub(dx, mul(mul(Num(2.0), node), dy)), syy) for dx, dy in zip(dev_x, dev_y)]
    # movcorr: r = Sxy / sqrt(Sxx Syy)
    root = Call("sqrt", (mul(sxx, syy),))
    return ([div(sub(dy, mul(mul(node, div(syy, root)), dx)), root) for dx, dy in zip(dev_x, dev_y)]
            + [div(sub(dx, mul(mul(node, div(sxx, root)), dy)), root) for dx, dy in zip(dev_x, dev_y)])
