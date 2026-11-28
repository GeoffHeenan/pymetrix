"""Task pages ("How do I…"): one task each, in the app and in code, with how to check the result and what goes wrong.

The pages are the Markdown files in this folder, shipped with the package. They are the one copy: GitHub shows them
as they are (``docs/how-do-i/README.md`` lists them), and the help shows each as a topic (``howto-<file name>``),
with an index topic, ``how-do-i``, that Help → How do I… opens. The help contents list them in a section of their
own, which is how the command palette (Ctrl+K) finds them.

Each file starts with front matter (``summary``, ``audience``, ``order``, and ``plugin`` for a page whose task needs a
plugin, such as ``pymetrix-eviews``: the index says so, and the page is shown whether the plugin is installed or not,
so that a reader learns it exists), then a ``# Title`` and the sections of the
template in ``docs/how-do-i/README.md``. For the help, links between the pages become help links, ``pymetrix:`` links
stay as they are, and links to the repository's documents (which the installed package does not have) become their
text. A fenced block whose info string is ``text <file name>`` shows a file the page's code reads; the tests write
it out before running the page's code (``tests/test_how_do_i.py``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

__all__ = ["INDEX", "INDEX_TITLE", "Page", "add_topics", "dialogs", "help_text", "index_text", "is_howto", "pages", "pages_for_dialog",
           "section"]

FOLDER = Path(__file__).resolve().parent
INDEX = "how-do-i"  # the index topic's id
INDEX_TITLE = "How do I…"
PREFIX = "howto-"  # each page's topic id: this and its file name


@dataclass(frozen=True)
class Page:
    """One task page."""

    id: str  # its help topic: "howto-estimate-and-test-an-equation"
    title: str  # its "# Title"
    summary: str  # one line for the index
    audience: str  # who it is written for
    order: int  # its place in the index
    path: Path
    text: str  # the Markdown after the front matter, its title included
    plugin: str = ""  # the plugin the task needs ("pymetrix-eviews"), if any


def pages() -> list[Page]:
    """The pages in this folder, in their order."""
    from ..help import _front_matter

    out = []
    for path in sorted(FOLDER.glob("*.md")):
        meta, text = _front_matter(path.read_text(encoding="utf-8"))
        heading = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        title = heading.group(1).strip() if heading else path.stem.replace("-", " ")
        out.append(Page(PREFIX + path.stem, title, str(meta.get("summary", "")), str(meta.get("audience", "")),
                        int(meta.get("order", 99)), path, text, str(meta.get("plugin", ""))))
    return sorted(out, key=lambda p: (p.order, p.title))


# dialogs a page opens only to have data to work on: their help does not list every page that starts with them
SETUP_DIALOGS = frozenset({"LoadExample"})


def dialogs(page: Page) -> tuple[str, ...]:
    """The dialogs (command class names) a page's In the app section opens, by the menu paths it writes."""
    from ..commands import COMMANDS
    from ..commands.menus import menu_order

    found = re.search(r"^## In the app\n(.*?)(?=^## |\Z)", page.text, re.MULTILINE | re.DOTALL)
    steps = found.group(1) if found else ""
    used = [info.cls.__name__ for info, path in menu_order(list(COMMANDS.values())) if path in steps]
    return tuple(dict.fromkeys(used))


def pages_for_dialog(command: str) -> list[Page]:
    """The pages whose steps in the app open the dialog ``command`` (a command class name): its help lists them."""
    if command in SETUP_DIALOGS:
        return []
    return [p for p in pages() if command in dialogs(p)]


def is_howto(topic_id: str) -> bool:
    return topic_id == INDEX or topic_id.startswith(PREFIX)


_LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]+)\)")


def help_text(page: Page) -> str:
    """The page as the help shows it: without its title (the topic gives it), its links as the help can follow."""
    from ..help import link

    body = re.sub(r"\A\s*#\s+.+\n+", "", page.text)

    def fix(m: re.Match) -> str:
        label, target = m.group(1), m.group(2)
        if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.IGNORECASE) or target.startswith("#"):
            return m.group(0)  # pymetrix:, https: and anchors as they are
        file = (page.path.parent / target.split("#")[0]).resolve()
        if file.parent == FOLDER and file.suffix == ".md":
            return f"[{label}]({link('topic', PREFIX + file.stem)})"
        return label  # a document of the repository: the installed package does not have it

    return _LINK.sub(fix, body).strip() + "\n"


def index_text() -> str:
    """The index topic: each page, with what it does and who it is for."""
    from ..help import link

    lines = ["Pages that each take one task from start to finish: the steps in the app (the menu paths and the fields "
             "as the dialogs show them), the same in code, how to check the result, and what goes wrong.", ""]
    for p in pages():
        lines.append(f"- [{p.title}]({link('topic', p.id)}): {p.summary}" + (f" *For {p.audience}.*" if p.audience else "")
                     + (f" Needs the {p.plugin} plugin." if p.plugin else ""))
    lines += ["", f"Also, for users of other packages: [Coming from EViews]({link('topic', 'coming-from-eviews')}) and "
                  f"[Coming from MATLAB/IRIS and Dynare]({link('topic', 'coming-from-iris')})."]
    return "\n".join(lines)


def section() -> tuple[str, list[tuple[str, str]]]:
    """The help contents' section of these pages: the index first, then the pages in order."""
    from ..help import link

    return INDEX_TITLE, [(INDEX_TITLE, link("topic", INDEX))] + [(p.title, link("topic", p.id)) for p in pages()]


def add_topics() -> None:
    """The index and every page as help topics."""
    from ..help import add_topic

    add_topic(INDEX, INDEX_TITLE, index_text(), (), ())
    for p in pages():
        add_topic(p.id, p.title, help_text(p), (), ())
