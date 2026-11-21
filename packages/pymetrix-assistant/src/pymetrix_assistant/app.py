"""The Assistant panel, two tabs:

- **Ask**: ask Claude about the project. What is sent is the project's choice (``pymetrix.disclosure.MODES``), off
  until the user chooses; the panel asks the first time. Answers come with proposals: **Open…** opens the step's
  dialog filled in (the user runs it with OK, or not) and **Skip** drops it; the step run is marked "proposed" in
  History (``Session.proposing``). **Record…** shows the assistant's record kept in the project (``audit``). The API
  key is kept in the system keychain.
- **Connect**: Allow connections, the address to give an assistant of the user's own, and the steps it has run.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from . import audit
from .chat import ToolError, show_text
from .server import AppServer, MainThread, serve

__all__ = ["create_panel", "local_full_context", "project_mode", "set_local_full_context", "set_project_mode"]

HELP = ("Tick to let an AI assistant on this computer (Claude Code, VS Code, another MCP client) work in this window. "
        "Give it the address below. Its steps run as yours do: one at a time, in History, undoable. It cannot run "
        "Python, write files or fetch data from the network. Untick to stop it.")

ASK_HELP = ("Ask Claude what to do next. It proposes steps; each opens in its dialog, filled in, and runs only when "
            "you press OK. What it is told about this project is chosen below.")

ATTACHMENT = audit.ATTACHMENT  # the project's choice of mode, and the assistant's record, are saved with it
PROPOSER = "Claude"  # how History and the script name who proposed a step

#: The modes offered here (step 2): the others in ``pymetrix.disclosure.MODES`` come later.
MODE_CHOICES = (
    ("off", "Off: nothing is sent"),
    ("metadata_only", "Metadata only, to Anthropic"),
    ("local_model", "Metadata only, to a model on this computer"),
    ("full_context", "Full context, to Anthropic: you approve each send"),
)

FULL_CONTEXT = ("Full context sends Anthropic more than metadata: with each question, the code of the project's steps "
                "(formulas, settings, file paths, typed code); and when the assistant asks to see one of an object's "
                "tables (estimates, tests, data), that table.\n\nEach send is shown to you first, and nothing goes "
                "unless you press Send. Use full context for this project?")


def confirm_send(parent: Any, title: str, note: str, text: str) -> bool:
    """Show ``text`` and ask whether to send it (full context): True only for Send."""
    from PySide6 import QtWidgets

    box = QtWidgets.QDialog(parent)
    box.setObjectName("assistant-confirm-send")
    box.setWindowTitle(title)
    lay = QtWidgets.QVBoxLayout(box)
    label = QtWidgets.QLabel(note)
    label.setWordWrap(True)
    lay.addWidget(label)
    body = QtWidgets.QPlainTextEdit(text)
    body.setReadOnly(True)
    lay.addWidget(body)
    buttons = QtWidgets.QDialogButtonBox()
    buttons.addButton("Send", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
    buttons.addButton("Don't send", QtWidgets.QDialogButtonBox.ButtonRole.RejectRole)
    buttons.accepted.connect(box.accept)
    buttons.rejected.connect(box.reject)
    lay.addWidget(buttons)
    box.resize(640, 480)
    return box.exec() == QtWidgets.QDialog.DialogCode.Accepted


def confirm_full_context(parent: Any) -> bool:
    from PySide6 import QtWidgets

    return QtWidgets.QMessageBox.question(parent, "Assistant", FULL_CONTEXT) == QtWidgets.QMessageBox.StandardButton.Yes

FIRST_USE = ("This project has not chosen where the assistant's questions go.\n\n"
             "With each question it sends the project's metadata only: the names, kinds, sizes and periods of its "
             "objects and the names of the steps taken. It never sends data values, estimates, formulas, labels, "
             "notes, report text, file paths or typed code. Press \"What is sent…\" to see it.\n\n"
             "Send it to Anthropic (Claude, with your API key), or to a model on this computer (Ollama or LM Studio, "
             "nothing leaves the machine)?")

LOCAL_URL_KEY = "assistant/local_url"  # this computer's model server and model, kept in the app's settings
LOCAL_MODEL_KEY = "assistant/local_model"


def ask_first_use(parent: Any) -> str | None:
    """The mode the user picks the first time a project asks, or None (cancelled)."""
    from PySide6 import QtWidgets

    box = QtWidgets.QMessageBox(parent)
    box.setWindowTitle("Assistant")
    box.setText(FIRST_USE)
    anthropic = box.addButton("Anthropic", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
    local = box.addButton("This computer", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
    box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
    box.exec()
    clicked = box.clickedButton()
    return "metadata_only" if clicked is anthropic else "local_model" if clicked is local else None


def course_rule() -> tuple[str, str]:
    """What the loaded courses allow an assistant (``pymetrix.help.Course.assistant``): ``("explain", course)``,
    ``("off", course)`` or ``("", "")``; the strictest applies."""
    from pymetrix import help as pm_help

    order = {"": 0, "explain": 1, "off": 2}
    rule, name = "", ""
    for course in pm_help.courses():
        said = getattr(course, "assistant", "") or ""
        if order.get(said, 0) > order[rule]:
            rule, name = said, course.name
    return rule, name


def project_mode(session: Any) -> str:
    """The project's mode: ``off`` until chosen."""
    from pymetrix.disclosure import DEFAULT_MODE, MODES

    raw = (getattr(session, "attachments", {}) or {}).get(ATTACHMENT, {}).get("mode", b"")
    mode = raw.decode("ascii", "replace") if isinstance(raw, bytes) else str(raw)
    return mode if mode in MODES else DEFAULT_MODE


def set_project_mode(session: Any, mode: str) -> None:
    from pymetrix.disclosure import MODES

    if mode not in MODES:
        raise ValueError(f"no mode {mode!r}; the modes are {', '.join(MODES)}")
    if project_mode(session) == mode and ATTACHMENT in session.attachments:
        return
    session.attachments.setdefault(ATTACHMENT, {})["mode"] = mode.encode("ascii")
    session.dirty = True


def local_full_context(session: Any) -> bool:
    """Whether a model on this computer is sent the project's full context (the project's choice; off until chosen)."""
    return (getattr(session, "attachments", {}) or {}).get(ATTACHMENT, {}).get("local_full", b"") == b"1"


def set_local_full_context(session: Any, on: bool) -> None:
    if local_full_context(session) == bool(on):
        return
    session.attachments.setdefault(ATTACHMENT, {})["local_full"] = b"1" if on else b"0"
    session.dirty = True


def create_panel(app: Any, client_factory: Callable[[], Any] | None = None,
                 local_request: Callable[..., Any] | None = None) -> Any:
    """The panel's widget (``AppPanel.create``): ``app`` is pymetrix's ``AppInterface``. ``client_factory`` makes
    Anthropic's client (default: with the keychain's key); ``local_request`` sends a request to the local model's
    server (default: over HTTP, on this computer only)."""
    from PySide6 import QtWidgets

    panel = QtWidgets.QWidget()
    panel.setObjectName("assistant-panel")
    outer = QtWidgets.QVBoxLayout(panel)
    outer.setContentsMargins(0, 0, 0, 0)
    tabs = QtWidgets.QTabWidget()
    outer.addWidget(tabs)
    ask_tab = _ask_tab(app, client_factory, local_request)
    tabs.addTab(ask_tab, "Ask")
    connect_tab, server_of, allow = _connect_tab(app)
    tabs.addTab(connect_tab, "Connect")
    panel.server = server_of  # for tests
    panel.allow = allow
    panel.ask = ask_tab
    panel.tabs = tabs
    return panel


# --- Ask -----------------------------------------------------------------------------------------------------------

def _ask_tab(app: Any, client_factory: Callable[[], Any] | None, local_request: Callable[..., Any] | None) -> Any:
    from PySide6 import QtCore, QtWidgets

    from pymetrix.disclosure import describe_session, to_text

    from .chat import (
        Conversation,
        api_key,
        client_from_keychain,
        forget_api_key,
        save_api_key,
    )
    from .local import DEFAULT_URL, LocalConversation, LocalModelError, list_models

    class _Relay(QtCore.QObject):
        done = QtCore.Signal(object)

        def refresh(self) -> None:  # a Qt object's method: the app stops calling it once the panel is gone
            show_mode()

    tab = QtWidgets.QWidget()
    tab.setObjectName("assistant-ask")
    layout = QtWidgets.QVBoxLayout(tab)
    intro = QtWidgets.QLabel(ASK_HELP)
    intro.setWordWrap(True)
    layout.addWidget(intro)
    course_note = QtWidgets.QLabel()
    course_note.setObjectName("assistant-course")
    course_note.setWordWrap(True)
    course_note.setStyleSheet("font-weight: bold")
    layout.addWidget(course_note)

    row = QtWidgets.QHBoxLayout()
    row.addWidget(QtWidgets.QLabel("Send:"))
    mode = QtWidgets.QComboBox()
    mode.setObjectName("assistant-mode")
    for key, text in MODE_CHOICES:
        mode.addItem(text, key)
    row.addWidget(mode, 1)
    record = QtWidgets.QPushButton("Record…")
    record.setToolTip("Every question asked in this project, what was sent with it, and what became of each proposal")
    shown = QtWidgets.QPushButton("What is sent…")
    shown.setToolTip("Show exactly what this project's question would carry")
    row.addWidget(shown)
    row.addWidget(record)
    layout.addLayout(row)

    key_row = QtWidgets.QHBoxLayout()
    key_label = QtWidgets.QLabel()
    key_label.setObjectName("assistant-key-state")
    key_row.addWidget(key_label, 1)
    set_key = QtWidgets.QPushButton("API key…")
    set_key.setToolTip("Your Anthropic API key, kept in this computer's keychain")
    key_row.addWidget(set_key)
    key_box = QtWidgets.QWidget()
    key_box.setLayout(key_row)
    key_row.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(key_box)

    # the app's own settings file (QSettings() alone names no organisation and writes elsewhere)
    settings = QtCore.QSettings(QtCore.QSettings.defaultFormat(), QtCore.QSettings.Scope.UserScope, "pymetrix", "desktop")
    local_box = QtWidgets.QWidget()
    local_box.setObjectName("assistant-local")
    local_row = QtWidgets.QHBoxLayout(local_box)
    local_row.setContentsMargins(0, 0, 0, 0)
    local_url = QtWidgets.QLineEdit(str(settings.value(LOCAL_URL_KEY, DEFAULT_URL) or DEFAULT_URL))
    local_url.setObjectName("assistant-local-url")
    local_url.setToolTip("The model server on this computer: Ollama http://localhost:11434/v1, "
                         "LM Studio http://localhost:1234/v1")
    local_model = QtWidgets.QComboBox()
    local_model.setObjectName("assistant-local-model")
    local_model.setEditable(True)
    local_model.setMinimumContentsLength(12)
    saved_model = str(settings.value(LOCAL_MODEL_KEY, "") or "")
    if saved_model:
        local_model.addItem(saved_model)
    local_model.lineEdit().setPlaceholderText("model, e.g. qwen3:8b")
    find = QtWidgets.QPushButton("Find models")
    find.setToolTip("List the models the server has")
    local_row.addWidget(local_url, 2)
    local_row.addWidget(local_model, 1)
    local_row.addWidget(find)
    layout.addWidget(local_box)
    local_full = QtWidgets.QCheckBox("Full context (it stays on this computer)")
    local_full.setObjectName("assistant-local-full")
    local_full.setToolTip("Also send the local model the code of the project's steps, and let it read an object's "
                          "tables (estimates, tests, data) when it asks. Nothing leaves this computer, so it is not "
                          "asked about each time; the record keeps what was sent.")
    layout.addWidget(local_full)

    transcript = QtWidgets.QTextBrowser()
    transcript.setObjectName("assistant-transcript")
    transcript.setOpenExternalLinks(False)

    # the proposals below the conversation, in a split the user can drag, and scrolling: each card keeps the height
    # its wrapped words need (squeezed into what was left of the panel, a card's reason was cut off)
    proposals_box = QtWidgets.QWidget()
    proposals_box.setObjectName("assistant-proposals")
    proposals_layout = QtWidgets.QVBoxLayout(proposals_box)
    proposals_layout.setContentsMargins(0, 0, 0, 0)
    proposals_layout.addStretch(1)
    proposals_scroll = QtWidgets.QScrollArea()
    proposals_scroll.setObjectName("assistant-proposals-scroll")
    proposals_scroll.setWidgetResizable(True)
    proposals_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    proposals_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    proposals_scroll.setWidget(proposals_box)
    proposals_scroll.hide()  # shown when there are proposals
    split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
    split.setObjectName("assistant-split")
    split.setChildrenCollapsible(False)
    split.addWidget(transcript)
    split.addWidget(proposals_scroll)
    split.setStretchFactor(0, 1)
    split.setStretchFactor(1, 1)
    layout.addWidget(split, 1)

    def fit_cards() -> None:
        """Each card's words as tall as they wrap at the width they have now: inside a scroll area Qt sizes a wrapped
        label for a narrower width, too tall, and squeezed outside one it was cut off."""
        for label in proposals_box.findChildren(QtWidgets.QLabel, "assistant-proposal-why"):
            if label.width() > 0:
                label.setFixedHeight(label.heightForWidth(label.width()))

    class _Refit(QtCore.QObject):
        def eventFilter(self, obj: Any, event: Any) -> bool:
            if event.type() == QtCore.QEvent.Type.Resize:
                QtCore.QTimer.singleShot(0, fit_cards)
            return False

    refit = _Refit(tab)
    proposals_scroll.viewport().installEventFilter(refit)

    question = QtWidgets.QPlainTextEdit()
    question.setObjectName("assistant-question")
    question.setPlaceholderText("Ask about this project, e.g. \"fit a consumption function on data\"")
    question.setMaximumHeight(80)
    layout.addWidget(question)
    buttons = QtWidgets.QHBoxLayout()
    new = QtWidgets.QPushButton("New conversation")
    resume = QtWidgets.QPushButton("Continue where I left off")
    resume.setObjectName("assistant-resume")
    resume.setToolTip("Bring back this project's last conversation from its record, so the assistant has it too")
    resume.hide()
    ask = QtWidgets.QPushButton("Ask")
    ask.setObjectName("assistant-ask-button")
    ask.setDefault(True)
    buttons.addWidget(new)
    buttons.addWidget(resume)
    buttons.addStretch(1)
    buttons.addWidget(ask)
    layout.addLayout(buttons)

    relay = _Relay(tab)
    state: dict[str, Any] = {"conversation": None, "busy": False, "worker": None, "session": None}

    def show_course() -> str:
        rule, name = course_rule()
        course_note.setVisible(bool(rule))
        course_note.setText({"explain": f"The course {name} asks the assistant to explain only: it proposes no steps.",
                             "off": f"The course {name} turns the assistant off."}.get(rule, ""))
        ask.setEnabled(rule != "off" and not state["busy"])
        question.setEnabled(rule != "off")
        return rule

    def destination() -> tuple[str, str, bool]:
        """Where questions go now: the provider, the mode, and whether full context is sent."""
        current = project_mode(app.session)
        provider = "local" if current == "local_model" else "anthropic"
        full = current == "full_context" or (current == "local_model" and local_full_context(app.session))
        return provider, current, full

    def show_resume() -> None:
        """Offer the project's last conversation, when there is none on screen and it can go where questions go now."""
        if state["conversation"] is not None or state.get("seed") or project_mode(app.session) == "off":
            resume.hide()
            return
        _, entries = audit.resumable(audit.read(app.session), *destination())
        resume.setVisible(bool(entries))
        if entries:
            resume.setText(f"Continue where I left off ({len(entries)} question{'s' if len(entries) != 1 else ''})")

    def show_mode() -> None:
        if state.get("session") is not app.session:  # another project: its own conversation, none yet
            state.update(session=app.session, conversation=None, seed=None, conversation_no=None)
            transcript.clear()
            clear_proposals()
        show_course()
        current = project_mode(app.session)
        i = mode.findData(current)
        mode.blockSignals(True)
        mode.setCurrentIndex(max(i, 0))
        mode.blockSignals(False)
        show_provider()
        show_resume()

    def show_provider() -> None:
        local = mode.currentData() == "local_model"
        local_box.setVisible(local)
        local_full.setVisible(local)
        local_full.blockSignals(True)
        local_full.setChecked(local_full_context(app.session))
        local_full.blockSignals(False)
        key_box.setVisible(not local)

    def mode_chosen(_index: int) -> None:
        if mode.currentData() == "full_context" and project_mode(app.session) != "full_context" \
                and not confirm_full_context(tab):
            show_mode()  # declined: as it was
            return
        set_project_mode(app.session, mode.currentData())
        show_provider()
        app.status({"off": "The assistant sends nothing",
                    "local_model": "The assistant sends this project's metadata only, to a model on this computer",
                    "full_context": "The assistant may send more, to Anthropic: you approve each send"}
                   .get(mode.currentData(), "The assistant sends this project's metadata only, to Anthropic"))

    def find_models() -> None:
        try:
            names = list_models(local_url.text(), local_request)
        except LocalModelError as exc:
            QtWidgets.QMessageBox.warning(tab, "Assistant", str(exc))
            return
        current = local_model.currentText()
        local_model.clear()
        local_model.addItems(names)
        if current in names:
            local_model.setCurrentText(current)
        app.status(f"{len(names)} model(s) on this computer" if names else "The server has no models: pull one first")

    def remember_local() -> None:
        settings.setValue(LOCAL_URL_KEY, local_url.text().strip())
        settings.setValue(LOCAL_MODEL_KEY, local_model.currentText().strip())

    def show_key() -> None:
        key_label.setText("API key: in the keychain" if api_key() else "API key: not set")

    def sends_full() -> bool:
        current = project_mode(app.session)
        return current == "full_context" or (current == "local_model" and local_full_context(app.session))

    def facts() -> str:
        full = sends_full()
        return to_text(describe_session(app.session, "full_context" if full else "metadata_only"))

    def show_sent() -> None:
        if project_mode(app.session) == "local_model" and sends_full():
            note = ("With each question, the model on this computer is sent this and your question. It may also "
                    "read an object's tables when it asks. Nothing leaves this computer.")
        elif project_mode(app.session) == "full_context":
            note = ("With each question, in full context, the assistant is sent this and your question, once you "
                    "press Send. It may also ask to see an object's tables: each is shown to you first.")
        else:
            note = ("With each question, in metadata-only mode, the assistant is sent this (and your question). "
                    "Nothing else about the project is sent.")
        _text_dialog(tab, "What is sent", note, facts())

    main_thread = MainThread()  # show_object's requests come from the question's thread: asked here, on the main one

    def show_on_main(name: str, section: str) -> str:
        session = app.session
        if name not in session.objects:
            raise ToolError(f"no object {name!r}; the project holds {', '.join(session.objects) or 'nothing yet'}")
        text = show_text(name, session.objects[name], section)
        local = project_mode(session) == "local_model"  # nothing leaves this computer: not asked each time
        sent = True if local else confirm_send(tab, "The assistant asks to see a table",
                            f"The assistant asks to see {name}" + (f" ({section})" if section else "") + ". Send this "
                            "to Anthropic?", text)
        audit.append(session, "shown", question=state.get("q"), object=name, section=section, sent=sent,
                     text=text if sent else "")
        if not sent:
            raise ToolError(f"The user declined to send {name}" + (f" ({section})" if section else "") + ".")
        return text

    def show(name: str, section: str) -> str:
        return main_thread(lambda: show_on_main(name, section))

    def ask_key() -> None:
        key, ok = QtWidgets.QInputDialog.getText(
            tab, "Anthropic API key", "Your API key (from console.anthropic.com). It is kept in this computer's "
            "keychain, not in the project. Leave empty to remove it.", QtWidgets.QLineEdit.EchoMode.Password)
        if not ok:
            return
        try:
            if key.strip():
                save_api_key(key)
            else:
                forget_api_key()
        except Exception as exc:  # noqa: BLE001 - no keychain: say so
            QtWidgets.QMessageBox.warning(tab, "Assistant", f"The key could not be kept in the keychain: {exc}")
        show_key()
        state["conversation"] = None  # the next question uses the new key

    def add(who: str, text: str) -> None:
        from html import escape

        body = escape(text).replace("\n", "<br>")
        transcript.append(f"<p><b>{escape(who)}:</b> {body}</p>")

    def clear_proposals() -> None:
        for card in proposals_box.findChildren(QtWidgets.QFrame, "assistant-proposal"):
            proposals_layout.removeWidget(card)
            card.deleteLater()
        proposals_scroll.hide()

    def add_proposal(p: Any, q: int, n: int) -> None:
        card = QtWidgets.QFrame()
        card.setObjectName("assistant-proposal")
        card.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        lay = QtWidgets.QVBoxLayout(card)
        card.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)  # its own height
        label = QtWidgets.QLabel(f"<b>{_escape(_dialog_name(p))}</b>" + (f": {_escape(p.why)}" if p.why else ""))
        label.setObjectName("assistant-proposal-why")
        label.setWordWrap(True)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        label.setToolTip(p.code)
        lay.addWidget(label)
        row = QtWidgets.QHBoxLayout()
        open_it = QtWidgets.QPushButton("Open…")
        open_it.setToolTip("Open its dialog, filled in: press OK there to run it")
        skip = QtWidgets.QPushButton("Skip")
        row.addStretch(1)
        row.addWidget(skip)
        row.addWidget(open_it)
        lay.addLayout(row)

        def opened() -> None:
            session = app.session
            before = len(session.history)
            with session.proposing(p.command_cls, PROPOSER):  # the step run from the dialog is marked as proposed
                app.open_dialog(p.command_cls, p.command)
            ran = [i for i, r in enumerate(session.history[before:], before + 1) if r.proposed_by and r.ok]
            changed = _changed(p.command, session.history[ran[0] - 1].command) if ran else []
            audit.append(session, "opened", question=q, proposal=n, command=p.name, step=ran[0] if ran else None,
                         changed=changed)
            if ran:
                label.setText(f"{label.text()} <i>(ran as step {ran[0]})</i>")
                fit_cards()
                open_it.setText("Open again…")
                skip.setEnabled(False)

        def skipped() -> None:
            audit.append(app.session, "skipped", question=q, proposal=n, command=p.name)
            card.hide()
            card.deleteLater()

        open_it.clicked.connect(opened)
        skip.clicked.connect(skipped)
        card.open_it, card.skip, card.proposal = open_it, skip, p  # for tests
        proposals_layout.insertWidget(proposals_layout.count() - 1, card)  # above the stretch
        if proposals_scroll.isHidden():  # the first card: half the room for the proposals (the divider moves)
            proposals_scroll.show()
            total = sum(split.sizes()) or split.height()
            split.setSizes([total * 2 // 5, total - total * 2 // 5])
        QtCore.QTimer.singleShot(0, fit_cards)

    def finished(outcome: Any) -> None:
        state["busy"] = False
        state["worker"] = None
        ask.setEnabled(True)
        ask.setText("Ask")
        text, sent, sent_mode = state.pop("pending", ("", "", ""))
        session = app.session
        q = audit.next_question(session)
        if isinstance(outcome, BaseException):
            add("Assistant", f"Could not ask: {_error_text(outcome)}")
            audit.append(session, "failed", question=q, text=text, mode=sent_mode, error=_error_text(outcome))
            return
        conversation = state["conversation"]
        fallback = bool(conversation is not None and conversation.fallback_used)
        provider = getattr(conversation, "provider", "anthropic")
        model_name = getattr(conversation, "model", "")
        if outcome.text:
            add("Claude" if provider == "anthropic" else model_name or "Model", outcome.text)
        where = {"url": conversation.url} if provider == "local" else {}
        audit.append(session, "asked", question=q, provider=provider, **where,
                     model=model_name, mode=sent_mode, text=text, sent=sent, reply=outcome.text,
                     fallback=fallback, explain=bool(getattr(conversation, "explain", False)),
                     full=bool(getattr(conversation, "full", False)), conversation=state.get("conversation_no"),
                     proposals=[{"n": n, "command": p.name, "code": p.code, "why": p.why}
                                for n, p in enumerate(outcome.proposals, 1)])
        for n, p in enumerate(outcome.proposals, 1):
            add_proposal(p, q, n)
        if fallback:
            app.status("Anthropic answered with its fallback model")

    relay.done.connect(finished)

    def confirm_mode() -> bool:
        if project_mode(app.session) != "off":
            return True
        chosen = ask_first_use(tab)
        if chosen is None:
            return False
        set_project_mode(app.session, chosen)
        show_mode()
        return True

    def asked() -> None:
        text = question.toPlainText().strip()
        if not text or state["busy"]:
            return
        rule = show_course()
        if rule == "off":
            return
        if not confirm_mode():
            return
        current = project_mode(app.session)
        explain = rule == "explain"
        full_local = current == "local_model" and local_full_context(app.session)
        wanted = (current, local_url.text().strip(), local_model.currentText().strip(), explain, full_local) \
            if current == "local_model" else (current, explain)
        if state["conversation"] is None or state.get("made_for") != wanted:  # a new provider or model: start afresh
            try:
                if current == "local_model":
                    state["conversation"] = LocalConversation(wanted[1], wanted[2], local_request, explain=explain,
                                                                   full=full_local, show=show)
                    remember_local()
                else:
                    state["conversation"] = Conversation((client_factory or client_from_keychain)(),
                                                         full=current == "full_context", show=show, explain=explain)
                state["made_for"] = wanted
                seed = state.pop("seed", None)
                if seed and seed[0] == destination():  # still going where the earlier answers came from
                    state["conversation"].seed(seed[1])
                else:
                    state["conversation_no"] = None
                if state.get("conversation_no") is None:
                    state["conversation_no"] = 1 + max((int(e.get("conversation", 0)) for e in audit.read(app.session)
                                                        if e.get("kind") == "asked"), default=0)
                resume.hide()
            except Exception as exc:  # noqa: BLE001 - no SDK, no key, no local model: say what to do
                add("Assistant", f"Could not start: {_error_text(exc)}")
                return
        conversation = state["conversation"]
        sent = facts()  # here, on the main thread, where the session lives
        if current == "full_context" and not confirm_send(tab, "Send to Anthropic?", "In full context, your question "
                                                          "goes with this. Send it?", sent):
            return  # nothing sent; the question stays in the box
        state["pending"] = (text, sent, project_mode(app.session))
        state["q"] = audit.next_question(app.session)
        add("You", text)
        question.clear()
        clear_proposals()
        state["busy"] = True
        ask.setEnabled(False)
        ask.setText("Thinking…")

        def work() -> None:
            try:
                outcome: Any = conversation.ask(text, sent)
            except Exception as exc:  # noqa: BLE001 - shown in the panel
                outcome = exc
            try:
                relay.done.emit(outcome)  # a queued connection: ``finished`` runs on the main thread
            except RuntimeError:  # the panel closed while the question was out
                pass

        state["worker"] = threading.Thread(target=work, name="pymetrix assistant question", daemon=True)
        state["worker"].start()

    def new_conversation() -> None:
        if state["busy"]:
            return
        state.update(conversation=None, seed=None, conversation_no=None)
        transcript.clear()
        clear_proposals()
        show_resume()

    def resume_conversation() -> None:
        """The last conversation back on screen, and given to the model with the next question."""
        if state["busy"]:
            return
        number, entries = audit.resumable(audit.read(app.session), *destination())
        if not entries:
            return
        transcript.clear()
        clear_proposals()
        provider = destination()[0]
        for e in entries:
            add("You", str(e.get("text") or ""))
            who = "Claude" if provider == "anthropic" else str(e.get("model") or "Model")
            answer = str(e.get("reply") or "")
            proposed = "".join(f"\n• {p.get('command')}: {p.get('why', '')}" for p in e.get("proposals", []))
            add(who, answer + (f"\n\nProposed:{proposed}" if proposed else ""))
        transcript.append("<p><i>Continued from this project's record. Earlier proposals are listed above; ask "
                          "again to have one as a card.</i></p>")
        state.update(conversation=None, seed=(destination(), audit.as_turns(entries)), conversation_no=number)
        resume.hide()

    mode.currentIndexChanged.connect(mode_chosen)
    shown.clicked.connect(show_sent)
    record.clicked.connect(lambda: _text_dialog(tab, "The assistant's record", "Kept in the project: each question, "
                                                "what was sent with it, the answer and what became of each proposal.",
                                                audit.as_text(audit.read(app.session))))
    set_key.clicked.connect(ask_key)
    ask.clicked.connect(asked)
    new.clicked.connect(new_conversation)
    resume.clicked.connect(resume_conversation)
    mode.currentIndexChanged.connect(lambda _i: show_resume())
    app.on_history_changed(relay.refresh)  # another project opened: its own choice
    find.clicked.connect(find_models)
    local_full.toggled.connect(lambda on: (set_local_full_context(app.session, on), show_resume()))
    show_mode()
    show_key()
    tab.question, tab.ask_button, tab.mode, tab.transcript, tab.proposals = question, ask, mode, transcript, proposals_box
    tab.worker = lambda: state["worker"]
    tab.local_url, tab.local_model, tab.find_models, tab.local_full = local_url, local_model, find, local_full
    tab.resume, tab.new_conversation = resume, new
    tab.confirm_mode = confirm_mode
    return tab


def _changed(proposed: Any, ran: Any) -> list[str]:
    """The settings the user changed in the dialog before running the proposed step."""
    from dataclasses import fields

    try:
        return [f.name for f in fields(proposed) if f.init and getattr(proposed, f.name, None) != getattr(ran, f.name, None)]
    except TypeError:  # not a dataclass
        return []


def _text_dialog(parent: Any, title: str, note: str, text: str) -> None:
    from PySide6 import QtWidgets

    box = QtWidgets.QDialog(parent)
    box.setObjectName("assistant-text")
    box.setWindowTitle(title)
    lay = QtWidgets.QVBoxLayout(box)
    label = QtWidgets.QLabel(note)
    label.setWordWrap(True)
    lay.addWidget(label)
    body = QtWidgets.QPlainTextEdit(text)
    body.setReadOnly(True)
    lay.addWidget(body)
    close = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
    close.rejected.connect(box.reject)
    lay.addWidget(close)
    box.resize(620, 460)
    box.exec()


def _dialog_name(p: Any) -> str:
    """The step as its dialog is called in the menus ("Equation"), else its class name."""
    try:
        from pymetrix.wording import menu_label

        menu = getattr(p.command_cls, "label", "") or getattr(p.command_cls, "menu", "") or ""
        return menu_label(menu).rstrip(".… ").split("/")[-1] or p.name
    except Exception:  # noqa: BLE001 - the class name will do
        return p.name


def _escape(text: str) -> str:
    from html import escape

    return escape(text)


def _error_text(exc: BaseException) -> str:
    name = type(exc).__name__
    if name == "LocalModelError":  # it says what to do
        return str(exc)
    if name == "AuthenticationError" or "api_key" in str(exc).lower():
        return "no valid Anthropic API key: press \"API key…\" to set one"
    if name == "ModuleNotFoundError":
        return "Anthropic's client is not installed: pip install anthropic"
    if name in ("APIConnectionError", "APITimeoutError"):
        return "Anthropic's API could not be reached; check the connection"
    return f"{name}: {exc}"


# --- Connect -------------------------------------------------------------------------------------------------------

def _connect_tab(app: Any) -> tuple[Any, Callable[[], AppServer | None], Any]:
    from PySide6 import QtWidgets

    tab = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(tab)
    intro = QtWidgets.QLabel(HELP)
    intro.setWordWrap(True)
    layout.addWidget(intro)
    allow = QtWidgets.QCheckBox("Allow connections")
    allow.setObjectName("assistant-allow")
    layout.addWidget(allow)
    address = QtWidgets.QLineEdit()
    address.setObjectName("assistant-address")
    address.setReadOnly(True)
    address.setPlaceholderText("not running")
    layout.addWidget(address)
    row = QtWidgets.QHBoxLayout()
    copy_url = QtWidgets.QPushButton("Copy address")
    copy_command = QtWidgets.QPushButton("Copy Claude Code command")
    copy_command.setToolTip("claude mcp add --transport http pymetrix <address>")
    for b in (copy_url, copy_command):
        b.setEnabled(False)
        row.addWidget(b)
    layout.addLayout(row)
    layout.addWidget(QtWidgets.QLabel("Steps the assistant has run:"))
    steps = QtWidgets.QListWidget()
    steps.setObjectName("assistant-steps")
    layout.addWidget(steps)
    state: dict[str, AppServer | None] = {"server": None}

    def stop() -> None:
        server = state["server"]
        if server is not None:
            server.stop()
        state["server"] = None
        try:
            address.clear()
            for b in (copy_url, copy_command):
                b.setEnabled(False)
        except RuntimeError:  # the panel is being destroyed: only the server needed stopping
            pass

    def toggled(on: bool) -> None:
        if not on:
            stop()
            app.status("The assistant can no longer connect")
            return
        rule, name = course_rule()
        if rule:  # a connected assistant runs steps: not while a course asks for explaining only, or for no assistant
            allow.setChecked(False)
            QtWidgets.QMessageBox.information(tab, "Assistant", f"The course {name} does not allow an assistant to "
                                              "run steps, so connections are off while it is loaded.")
            return
        try:
            state["server"] = serve(app, on_step=steps.addItem)
        except Exception as exc:  # noqa: BLE001 - say why, and leave it off
            allow.setChecked(False)
            QtWidgets.QMessageBox.warning(tab, "Assistant", f"The assistant's server could not start: {exc}")
            return
        address.setText(state["server"].url)
        for b in (copy_url, copy_command):
            b.setEnabled(True)
        app.status("An assistant on this computer can now connect (see the Assistant panel)")

    def copy(text: str) -> None:
        QtWidgets.QApplication.clipboard().setText(text)
        app.status("Copied")

    allow.toggled.connect(toggled)
    copy_url.clicked.connect(lambda: copy(state["server"].url) if state["server"] else None)
    copy_command.clicked.connect(lambda: copy(state["server"].claude_code_command()) if state["server"] else None)
    tab.destroyed.connect(lambda *_: stop())  # the app closing stops the server
    return tab, lambda: state["server"], allow
