"""Write docs/defaults.md from the in-app help page ``defaults`` (Help → Help contents → Defaults, and other
packages' defaults), so the two say the same thing.

    python scripts/make_defaults_page.py              # docs/defaults.md
    python scripts/make_defaults_page.py <file>       # elsewhere

tests/test_defaults_page.py fails when docs/defaults.md differs from what this writes: change the help page
(``add_topic("defaults", ...)`` in src/pymetrix/help.py), then rerun this.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def defaults_markdown() -> str:
    """The page's Markdown, from the help topic."""
    from pymetrix.help import _TOPICS, topic

    t = topic("defaults")
    out = [f"# {t.title}", "",
           "<!-- Written by scripts/make_defaults_page.py from the in-app help page `defaults` (src/pymetrix/help.py): "
           "edit that, then rerun the script. -->", "",
           "The same page is in the app: **Help → Help contents**, *Defaults, and other packages' defaults*.", "", t.text]
    if t.see_also:
        out += ["", "## See also, in the app's help", ""]
        out += [f"- {_TOPICS[s].title if s in _TOPICS else s}" for s in t.see_also]
    if t.references:
        out += ["", "## References", ""] + [f"- {r}" for r in t.references]
    return "\n".join(out) + "\n"


def main(argv: list[str]) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    target = Path(argv[1]) if len(argv) > 1 else ROOT / "docs" / "defaults.md"
    target.write_text(defaults_markdown(), encoding="utf-8")
    print(f"-> {target}")


if __name__ == "__main__":
    main(sys.argv)
