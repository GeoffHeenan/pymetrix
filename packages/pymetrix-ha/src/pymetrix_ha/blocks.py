"""Blocks: sequence-jacobian's own decorators, with the checks its source-reading needs.

sequence-jacobian reads a block's output names from the source text of its last ``return`` line
(``inspect.getsource`` and a regular expression). Two things go wrong:

- **Blocks defined in text run by ``exec``** (a string, or code run without a file) have no source to read, and the
  decorator fails with "could not get source code". ``blocks_from_source`` runs such text with its source registered
  where ``inspect`` finds it, so the blocks work as if they were in a file; ``simple`` and ``het`` here say so when
  the library cannot read a block.
- **A ``return`` of an expression** (``return (1 + r) * c ** -1``) is read as an output with that text for a name,
  which no later block can use. ``check_blocks`` finds it before solving and names the block.
"""

from __future__ import annotations

import hashlib
import linecache
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from pymetrix.models.solvers import ModelError

from ._library import library, source_fingerprint

_NO_SOURCE = ("blocks are read from their source text, which this {what} has none of (it was defined in text run by "
              "exec, such as a string or a console without a file). Put the blocks in a .py file and import it, or "
              "run their text with pymetrix_ha.blocks_from_source(text)")


class BlockError(ModelError):
    """A block sequence-jacobian cannot use as written; the message names the block and what to write."""


def _no_source(exc: BaseException) -> bool:
    return isinstance(exc, (OSError, TypeError)) and "source" in str(exc)


def simple(f: Callable[..., Any]) -> Any:
    """``sequence_jacobian.simple``: aggregate equations as a block (``K(-1)`` for a lag, ``r(+1)`` for a lead). The
    outputs are the names on the function's last ``return`` line."""
    sj = library()
    try:
        block = sj.simple(f)
    except (OSError, TypeError) as exc:
        if _no_source(exc):
            raise BlockError(f"block {getattr(f, '__name__', f)!r}: " + _NO_SOURCE.format(what="function")) from exc
        raise
    check_blocks([block])
    return block


def het(exogenous: Any, policy: Any, backward: Any, backward_init: Callable[..., Any] | None = None,
        hetinputs: Iterable[Callable[..., Any]] | None = None, hetoutputs: Iterable[Callable[..., Any]] | None = None) -> Any:
    """``sequence_jacobian.het``: a household block from its backward step (one step of the endogenous grid method,
    say), with the exogenous Markov matrix, the policy and the backward variable named."""
    sj = library()

    def decorate(f: Callable[..., Any]) -> Any:
        try:
            block = sj.het(exogenous=exogenous, policy=policy, backward=backward, backward_init=backward_init,
                           hetinputs=hetinputs, hetoutputs=hetoutputs)(f)
        except (OSError, TypeError) as exc:
            if _no_source(exc):
                raise BlockError(f"block {getattr(f, '__name__', f)!r}: " + _NO_SOURCE.format(what="function")) from exc
            raise
        check_blocks([block])
        return block

    return decorate


def blocks_from_source(text: str, name: str = "blocks") -> dict[str, Any]:
    """Run Python text that defines blocks, with its source registered for ``inspect`` (as a file would be), and
    return what it defined: ``ns = blocks_from_source(text); HAModel([ns["household"], ns["firm"]], ...)``. This is
    how blocks typed into a console or kept as text in a project work despite sequence-jacobian reading blocks from
    their source. The text is run as code: only run text you trust."""
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    filename = f"<pymetrix-ha {name} {digest}>"
    lines = [line if line.endswith("\n") else line + "\n" for line in text.splitlines()]
    linecache.cache[filename] = (len(text), None, lines, filename)  # no modification time: never dropped as stale
    namespace: dict[str, Any] = {"__name__": f"pymetrix_ha_blocks_{digest}"}
    exec(compile(text, filename, "exec"), namespace)  # noqa: S102 - the user's own model code, run as asked
    return {k: v for k, v in namespace.items() if not k.startswith("__")}


def leaves(block: Any) -> list[Any]:
    """The simple, het and other single blocks inside ``block`` (combined and solved blocks opened), in order."""
    inner = getattr(block, "blocks", None)
    if isinstance(inner, (list, tuple)) and type(block).__name__ == "CombinedBlock":
        return [leaf for b in inner for leaf in leaves(b)]
    if type(block).__name__ == "SolvedBlock":
        return leaves(block.block)
    return [block]


def _functions(block: Any) -> list[Any]:
    """The library's wrapped functions (``ExtendedFunction``) that make up a single block."""
    kind = type(block).__name__
    if kind == "SimpleBlock":
        return [block.f]
    if kind == "HetBlock":
        out = [block.backward_fun]
        for combined in (block.hetinputs, block.hetoutputs):
            if combined is not None:
                out += list(combined.functions.values())
        if block.backward_init is not None:
            out.append(block.backward_init)
        return out
    return []


def check_blocks(blocks: Iterable[Any]) -> None:
    """Each block's outputs, as the library read them from its ``return`` lines, are names: raise ``BlockError``,
    naming the block and the function, when one is an expression (or a ``return`` split over lines)."""
    for block in blocks:
        for leaf in leaves(block):
            for fn in _functions(leaf):
                bad = [o for o in fn.outputs if not str(o).isidentifier()]
                if bad:
                    where = f"block {leaf.name!r}" + (f", function {fn.name!r}" if fn.name != leaf.name else "")
                    raise BlockError(
                        f"{where}: its outputs are read from the text of its last return line, which gives "
                        f"{', '.join(repr(str(o)) for o in bad)}, not names. Assign each output to a name and return "
                        "the names on one line, e.g. c = coh - a, then return Va, a, c")


def describe(block: Any) -> dict[str, Any]:
    """A single block for provenance: its name, kind, inputs and outputs, and a fingerprint of its source text."""
    fns = _functions(block)
    return {"name": str(block.name), "kind": type(block).__name__, "inputs": [str(i) for i in block.inputs],
            "outputs": [str(o) for o in block.outputs],
            "functions": [str(fn.name) for fn in fns],
            "source": source_fingerprint(*(fn.f for fn in fns)) if fns else ""}


def het_blocks(blocks: Iterable[Any]) -> list[Any]:
    """The het blocks among ``blocks`` (combined and solved blocks opened)."""
    return [leaf for b in blocks for leaf in leaves(b) if type(leaf).__name__ == "HetBlock"]


def find(blocks: Iterable[Any], name: str | Any) -> Any:
    """The single block called ``name`` (or ``name`` itself, if it is a block)."""
    if not isinstance(name, str):
        return name
    found = {str(leaf.name): leaf for b in blocks for leaf in leaves(b)}
    if name not in found:
        raise BlockError(f"no block {name!r}; the model's blocks are {', '.join(found)}")
    return found[name]


def check_inputs(blocks: Iterable[Any], given: Mapping[str, Any] | Iterable[str]) -> None:
    """Every block input is given (calibrated, an unknown, a shock) or another block's output: raise ``BlockError``
    naming the block and the input otherwise."""
    singles = [leaf for b in blocks for leaf in leaves(b)]
    made = {str(o) for leaf in singles for o in leaf.outputs}
    known = set(map(str, given)) | made
    for leaf in singles:
        missing = [str(i) for i in leaf.inputs if str(i) not in known]
        if missing:
            raise BlockError(f"block {leaf.name!r} needs {', '.join(repr(m) for m in missing)}, which no block makes and the "
                             "calibration, the steady state's unknowns and the shocks do not give")
