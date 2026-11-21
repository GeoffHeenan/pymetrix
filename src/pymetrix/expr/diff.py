"""Symbolic differentiation of expression trees.

Used for NLS gradients, model Jacobians and delta-method restrictions.
"""

from __future__ import annotations

from .functions import FUNCTIONS, partials
from .nodes import (
    COMPARISONS,
    LOGICAL,
    MARKERS,
    BinOp,
    Call,
    Coef,
    Expr,
    ExpressionError,
    Name,
    Neg,
    Num,
    add,
    div,
    mul,
    neg,
    power,
    sub,
)


def depends(node: Expr, wrt: Name | Coef) -> bool:
    return any(n == wrt for n in node.walk())


def diff(node: Expr, wrt: Name | Coef) -> Expr:
    """Derivative of ``node`` with respect to a variable-at-lag or a parameter."""
    return _diff(node, wrt, {})


def _depends(node: Expr, wrt: Name | Coef, seen: dict[int, tuple[Expr, bool]]) -> bool:
    """``depends``, remembered for each node of the tree being differentiated: asked again of each subtree on the way
    down, the plain walk is quadratic in the size of the tree (cubic for the Jacobian of a long sum). The node is kept
    with its answer, so its id cannot be reused by another node during the differentiation."""
    hit = seen.get(id(node))
    if hit is not None:
        return hit[1]
    found = node == wrt or any(_depends(child, wrt, seen) for child in node.children())
    seen[id(node)] = (node, found)
    return found


def _diff(node: Expr, wrt: Name | Coef, seen: dict[int, tuple[Expr, bool]]) -> Expr:
    if node == wrt:
        return Num(1.0)
    if not _depends(node, wrt, seen):
        return Num(0.0)
    if isinstance(node, Neg):
        return neg(_diff(node.operand, wrt, seen))
    if isinstance(node, BinOp) and (node.op in COMPARISONS or node.op in LOGICAL):
        return Num(0.0)  # a condition is flat wherever it is not jumping, as step() is
    if isinstance(node, BinOp):
        a, b = node.left, node.right
        da, db = _diff(a, wrt, seen), _diff(b, wrt, seen)
        if node.op == "+":
            return add(da, db)
        if node.op == "-":
            return sub(da, db)
        if node.op == "*":
            return add(mul(da, b), mul(a, db))
        if node.op == "/":
            return div(sub(mul(da, b), mul(a, db)), power(b, Num(2.0)))
        if _depends(b, wrt, seen):  # d(a^b) = a^b * (b' log a + b a'/a)
            return mul(node, add(mul(db, Call("log", (a,))), div(mul(b, da), a)))
        return mul(mul(b, power(a, sub(b, Num(1.0)))), da)
    if isinstance(node, Call):
        if node.fn in MARKERS:
            raise ExpressionError(f"{node.fn}() cannot be differentiated")
        if node.elementwise:  # the derivative of the argument that is larger (max) or smaller (min); at a tie, the first
            a, b = node.args
            first = Call("step", (sub(a, b) if node.fn == "max" else sub(b, a),))
            return add(mul(first, _diff(a, wrt, seen)), mul(sub(Num(1.0), first), _diff(b, wrt, seen)))
        if node.fn in FUNCTIONS:  # distributions, erf and moving windows: the chain rule over their arguments
            out: Expr = Num(0.0)
            for i, (arg, partial) in enumerate(zip(node.args, partials(node))):
                if not _depends(arg, wrt, seen):
                    continue
                if partial is None:
                    raise ExpressionError(_no_partial(node, i))
                out = add(out, mul(partial, _diff(arg, wrt, seen)))
            return out
        u = node.args[0]
        du = _diff(u, wrt, seen)
        if node.fn == "log":
            return div(du, u)
        if node.fn == "exp":
            return mul(node, du)
        if node.fn == "sqrt":
            return div(du, mul(Num(2.0), node))
        if node.fn == "abs":
            return mul(Call("sign", (u,)), du)
        if node.fn in ("sign", "step", "not"):
            return Num(0.0)
    raise ExpressionError(f"cannot differentiate {node}")


def _no_partial(node: Call, i: int) -> str:
    from .functions import DISTRIBUTIONS, EXPANDING, FAMILIES, MOVING

    if node.fn in EXPANDING:
        return (f"{node} cannot be differentiated: an expanding window uses every period up to t, not fixed lags, so it "
                "has no derivative and cannot be linearised; use a moving window (movavg(x, n)) or a derived series")
    if DISTRIBUTIONS.get(node.fn, ("", ""))[1] == "inv" and DISTRIBUTIONS[node.fn][0] in ("poiss", "bino"):
        return (f"{node} cannot be differentiated: {node.fn}() is a step function of its arguments (an integer quantile), "
                "so it has no derivative and cannot be linearised; make it a number, a parameter or a derived series")
    if node.fn in MOVING:
        return (f"{node} cannot be differentiated (a moving minimum or maximum jumps between periods); "
                "write it with max() or min() of the lags to linearise it")
    family = FAMILIES[DISTRIBUTIONS[node.fn][0]]
    which = family.params[i - 1]
    return (f"{node} cannot be differentiated with respect to its argument {which} ({node.args[i]}): no derivative "
            f"of {node.fn}() in {which} is offered, so it cannot be linearised in it; make {which} a number or a "
            "parameter that is not estimated")
