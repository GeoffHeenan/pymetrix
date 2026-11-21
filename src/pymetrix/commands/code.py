"""Typed code as a session step: what the Immediate and Code windows run.

Every cell is recorded, so the history, the script and projects reproduce a
session that mixes dialogs and code. Code that only IPython understands
(``%timeit``, ``x?``, ``!dir``) is recorded but cannot be replayed; it appears
in the script as a comment.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Literal

from .registry import Command, Widget, command

if TYPE_CHECKING:
    from .session import Session


def _python(code: str) -> ast.Module | None:
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


@command(menu=None)
@dataclass(frozen=True)
class RunCode(Command):
    """Python typed in the Immediate window or run from the Code window."""

    code: Annotated[str, Widget("text", help="Python code")]
    source: Annotated[Literal["immediate", "editor", "script"], Widget("choice", help="where the code was run from")] = "immediate"

    @property
    def replayable(self) -> bool:
        return _python(self.code) is not None

    def reads(self, session: Session, versions: dict[str, int]) -> set[str]:
        """Workspace objects the code uses: names it reads (or, for IPython syntax, mentions)."""
        tree = _python(self.code)
        if tree is None:
            used = set(re.findall(r"[A-Za-z_]\w*", self.code))
        else:
            used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
            used |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Del)}
            used |= {n.target.id for n in ast.walk(tree) if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name)}
        return used & set(versions)

    def run(self, session: Session) -> Any:
        tree = _python(self.code)
        if tree is None:  # IPython-only syntax: recorded, not replayed
            return None
        session.push()
        result = None
        body, last = (tree.body[:-1], tree.body[-1]) if tree.body and isinstance(tree.body[-1], ast.Expr) else (tree.body, None)
        try:
            exec(compile(ast.Module(body, []), f"<{self.source}>", "exec"), session.namespace)  # noqa: S102 - the user's own code
            if last is not None:
                result = eval(compile(ast.Expression(last.value), f"<{self.source}>", "eval"), session.namespace)  # noqa: S307
        finally:
            session.pull()
        return result

    def to_code(self) -> str:
        code = self.code.rstrip()
        if self.replayable:
            return code
        return "\n".join(f"# (IPython only, not replayed) {line}" for line in code.splitlines())
