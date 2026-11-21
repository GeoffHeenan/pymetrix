"""Help in the desktop UI: a viewer for help pages, the help browser, and course-material settings.

Pages come from ``pymetrix.help`` as Markdown with ``pymetrix:`` links, which
the viewer follows itself; web links and files (slides, notebooks) open in
the system's default application.
"""

from __future__ import annotations

from typing import Any

from .. import help as hp
from .messages import titled


def _qt() -> Any:
    from PySide6 import QtWidgets

    return QtWidgets


def _open_external(url: str) -> None:
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QDesktopServices

    QDesktopServices.openUrl(QUrl(url))


def help_view(parent: Any = None) -> Any:
    """A page viewer with Back and Contents buttons. ``view.go(link)`` shows a help link;
    ``view.show_markdown(text)`` shows generated text (such as a dialog's context help)."""
    Qt = _qt()
    view = Qt.QWidget(parent)
    layout = Qt.QVBoxLayout(view)
    layout.setContentsMargins(0, 0, 0, 0)
    bar = Qt.QHBoxLayout()
    back = Qt.QPushButton("◀ Back")
    home = Qt.QPushButton("Contents")
    bar.addWidget(back)
    bar.addWidget(home)
    bar.addStretch()
    layout.addLayout(bar)
    text = Qt.QTextBrowser()
    text.setOpenLinks(False)
    layout.addWidget(text, 1)
    history: list[tuple[str, str]] = []  # (link or "", markdown)
    on_change: list[Any] = []  # callbacks after a page is shown

    def render(markdown: str) -> None:
        text.setMarkdown(markdown)
        text.verticalScrollBar().setValue(0)
        back.setEnabled(len(history) > 1)
        for callback in on_change:
            callback()

    def show_markdown(markdown: str, key: str = "") -> None:
        history.append((key, markdown))
        render(markdown)

    def go(url: str) -> None:
        if not url.startswith(hp.SCHEME):
            _open_external(url)
            return
        try:
            markdown = hp.page_for_link(url)
        except KeyError as exc:
            markdown = f"# Not found\n\n{exc.args[0] if exc.args else exc}"
        show_markdown(markdown, url)

    def go_back() -> None:
        if len(history) > 1:
            history.pop()
            render(history[-1][1])

    text.anchorClicked.connect(lambda url: go(url.toString()))
    back.clicked.connect(go_back)
    home.clicked.connect(lambda: go(hp.link("home")))
    view.go, view.show_markdown, view.back, view.text, view.history, view.on_change = go, show_markdown, go_back, text, history, on_change
    view.current = lambda: history[-1][0] if history else ""
    return view


def help_window(parent: Any = None, start: str | None = None) -> Any:
    """The help browser: search and contents on the left, the page on the right."""
    Qt = _qt()
    from PySide6.QtCore import Qt as QtNS

    window = Qt.QWidget(parent, QtNS.WindowType.Window)
    window.setWindowTitle("PyMetrix help")
    from .fit import fit_to_screen

    fit_to_screen(window, 980, 680)
    split = Qt.QSplitter(window)
    left = Qt.QWidget()
    side = Qt.QVBoxLayout(left)
    side.setContentsMargins(0, 0, 0, 0)
    search = Qt.QLineEdit()
    search.setPlaceholderText("Search help, topics and course notes…")
    search.setClearButtonEnabled(True)
    tree = Qt.QTreeWidget()
    tree.setHeaderHidden(True)
    side.addWidget(search)
    side.addWidget(tree, 1)
    view = help_view()
    split.addWidget(left)
    split.addWidget(view)
    split.setSizes([280, 700])
    Qt.QVBoxLayout(window).addWidget(split)

    def fill_contents() -> None:
        tree.clear()
        for title, pages in hp.contents():
            section = Qt.QTreeWidgetItem([title])
            for page_title, url in pages:
                item = Qt.QTreeWidgetItem([page_title])
                item.setData(0, QtNS.ItemDataRole.UserRole, url)
                section.addChild(item)
            tree.addTopLevelItem(section)

    kinds = {"topic": "Topic", "note": "Course note", "notebook": "Example notebook", **hp.GROUP_TITLES}

    def run_search(words: str) -> None:
        if not words.strip():
            fill_contents()
            return
        tree.clear()
        for kind, title, obj in hp.search(words):
            item = Qt.QTreeWidgetItem([f"{title}  ({kinds.get(kind, kind)})"])
            item.setData(0, QtNS.ItemDataRole.UserRole, hp.link_of(obj))
            tree.addTopLevelItem(item)
        if not tree.topLevelItemCount():
            tree.addTopLevelItem(Qt.QTreeWidgetItem(["No matches"]))

    def open_item(item: Any) -> None:
        url = item.data(0, QtNS.ItemDataRole.UserRole)
        if url:
            view.go(url)
        else:
            item.setExpanded(not item.isExpanded())

    def follow_page() -> None:
        """The contents follow the page shown: its entry selected, its section open (a link followed in the text, or
        a dialog's F1, opened a page the tree did not show)."""
        url = view.current()
        if not url or search.text().strip():
            return
        for i in range(tree.topLevelItemCount()):
            section = tree.topLevelItem(i)
            for j in range(section.childCount()):
                item = section.child(j)
                if item.data(0, QtNS.ItemDataRole.UserRole) == url:
                    section.setExpanded(True)
                    tree.setCurrentItem(item)
                    tree.scrollToItem(item)
                    return
        tree.setCurrentItem(None)

    search.textChanged.connect(run_search)
    tree.itemClicked.connect(lambda item, _col: open_item(item))
    fill_contents()
    view.on_change.append(follow_page)
    view.go(start or hp.link("home"))
    window.view, window.search, window.tree, window.refresh = view, search, tree, lambda: run_search(search.text())
    return window


COURSES_KEY = "help/courses"


def restore_courses(settings: Any) -> list[str]:
    """Load the course folders remembered in the settings; returns problems to report."""
    problems = []
    for folder in _user_folders(settings):
        try:
            hp.load_course(folder)
        except (OSError, ValueError) as exc:
            problems.append(f"{folder}: {exc}")
    return problems


def _user_folders(settings: Any) -> list[str]:
    return [str(f) for f in (settings.value(COURSES_KEY, [], type=list) or [])]


def course_dialog(settings: Any, parent: Any = None, session: Any = None) -> Any:
    """Course material in the help: folders of lecture notes (.md) and example notebooks (.ipynb).

    A folder is either the user's own (remembered between sessions) or part of
    the current project (saved inside the .pymx file, so students who open the
    project get it); a folder can be both.
    """
    Qt = _qt()
    from pathlib import Path

    dialog = Qt.QDialog(parent)
    dialog.setWindowTitle("Course material")
    from .fit import fit_to_screen

    fit_to_screen(dialog, 720, 420)
    layout = Qt.QVBoxLayout(dialog)
    from .metrics import set_margins

    set_margins(layout, "dialog_margin")
    layout.addWidget(Qt.QLabel("Folders of lecture notes (.md) and example notebooks (.ipynb) shown in the help.\n"
                               "Notes link themselves to help pages with front matter: see the example in examples/course.\n"
                               "Include a folder in the project to save it inside the .pymx file with the data and commands."))
    listing = Qt.QListWidget()
    layout.addWidget(listing, 1)
    report = Qt.QPlainTextEdit()
    report.setReadOnly(True)
    report.setMaximumHeight(110)
    report.setPlaceholderText("Problems found in the course material are listed here.")
    layout.addWidget(report)
    in_project = (lambda: list(session.courses)) if session is not None else (lambda: [])

    def refresh() -> None:
        keep = listing.currentItem().data(256) if listing.currentItem() is not None else None
        listing.clear()
        mine, project = set(_user_folders(settings)), set(in_project())
        for course in hp.courses():
            folder = str(course.path)
            where = ", ".join(w for w, on in (("in project", folder in project), ("mine", folder in mine)) if on)
            item = Qt.QListWidgetItem(f"{course.name}  —  {len(course.notes)} notes  —  {where or 'loaded'}  —  {folder}")
            item.setData(256, folder)
            listing.addItem(item)
            if folder == keep:
                listing.setCurrentItem(item)
        report.setPlainText("\n".join(f"{c.name}: {p}" for c in hp.courses() for p in c.problems))

    def selected() -> str | None:
        item = listing.currentItem()
        return item.data(256) if item is not None else None

    def add_folder(folder: str) -> None:
        folder = str(Path(folder).expanduser().resolve())
        hp.load_course(folder)
        mine = _user_folders(settings)
        if folder not in mine:
            settings.setValue(COURSES_KEY, mine + [folder])
        refresh()

    def add() -> None:
        folder = Qt.QFileDialog.getExistingDirectory(dialog, "Choose a course folder")
        if folder:
            try:
                add_folder(folder)
            except (OSError, ValueError) as exc:
                Qt.QMessageBox.critical(dialog, *titled("Could not read the course folder", str(exc)))

    def remove() -> None:
        folder = selected()
        if folder is None:
            return
        settings.setValue(COURSES_KEY, [f for f in _user_folders(settings) if f != folder])
        if session is not None:
            session.detach_course(folder)
        hp.unload_course(folder)
        refresh()

    def include(on: bool) -> None:
        folder = selected()
        if folder is not None and session is not None:
            (session.attach_course if on else session.detach_course)(folder)
            refresh()

    def reload() -> None:
        for course in hp.courses():
            hp.load_course(course.path)
        refresh()

    buttons = Qt.QHBoxLayout()
    actions = [("Add folder…", add), ("Remove", remove), ("Reload", reload)]
    if session is not None:
        actions[1:1] = [("Include in project", lambda: include(True)), ("Leave out of project", lambda: include(False))]
    for label, handler in actions:
        button = Qt.QPushButton(label)
        button.clicked.connect(lambda _=False, h=handler: h())
        buttons.addWidget(button)
    buttons.addStretch()
    close = Qt.QPushButton("Close")
    close.clicked.connect(dialog.accept)
    buttons.addWidget(close)
    layout.addLayout(buttons)
    refresh()
    dialog.listing, dialog.report, dialog.add_folder, dialog.include, dialog.remove = listing, report, add_folder, include, remove
    return dialog


def forget_project_courses(old: Any, settings: Any) -> None:
    """When a session is replaced, take its project's courses out of the help unless they are the user's own."""
    mine = set(_user_folders(settings))
    for folder in getattr(old, "courses", []):
        if folder not in mine:
            hp.unload_course(folder)
