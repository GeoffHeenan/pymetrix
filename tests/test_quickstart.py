"""The model builders' quickstart (docs/modelling-quickstart.md) runs as written: its code blocks, in order, as one
script."""

import re
from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "docs" / "modelling-quickstart.md"


def test_the_quickstart_runs_as_written():
    blocks = re.findall(r"```python\n(.*?)```", DOC.read_text(encoding="utf-8"), flags=re.S)
    assert len(blocks) >= 6
    scope: dict = {"__name__": "quickstart"}
    for i, block in enumerate(blocks, 1):
        exec(compile(block, f"{DOC.name}, block {i}", "exec"), scope)
    assert scope["post"].draws.shape == (2, 300, 3)
