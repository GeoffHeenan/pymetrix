"""Make a tutorial's Jupyter notebook from its Markdown page (run from the repository root).

    python scripts/make_tutorial_notebook.py docs/tutorial.md      # writes docs/tutorial.ipynb (from packages/pymetrix-finance)

Each ```python block becomes a code cell and the text between them a Markdown cell, so the notebook and the page say
the same thing; tests/test_finance_polish.py checks that the committed notebook is what this script makes.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def notebook(markdown: str) -> dict:
    cells = []
    for i, part in enumerate(re.split(r"```python\n(.*?)```\n?", markdown, flags=re.S)):
        text = part.strip("\n")
        if not text:
            continue
        kind = "code" if i % 2 else "markdown"
        cell = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def write(md_path: Path) -> Path:
    out = md_path.with_suffix(".ipynb")
    out.write_text(json.dumps(notebook(md_path.read_text(encoding="utf-8")), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    print(write(Path(sys.argv[1] if len(sys.argv) > 1 else "docs/tutorial.md")))
