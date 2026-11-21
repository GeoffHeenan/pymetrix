"""Keys in the platform's own words, for text the app shows: tooltips, labels, the Keyboard shortcuts list.

Shortcuts are set in Qt's portable form ("Ctrl+K"), which Qt maps to the platform's keys: on a Mac its Ctrl is the
Command key. Text that names a key must follow, or a Mac's tooltips say "Ctrl+K" for a key that is ⌘K there.
``key_text`` gives what Qt's native text gives (``QKeySequence.toString(NativeText)``, as the menus show): "Ctrl+K" on
Linux and Windows, "⌘K" on a Mac.

``MAC`` says whether to give a Mac's words. It is ``sys.platform == "darwin"``; tests set it on another system to see
the Mac's text, which ``mac_text`` then makes the way Qt makes it on a Mac (on a Mac itself Qt's own text is used, and
tests/test_ui_key_text.py checks there that ``mac_text`` gives the same).
"""

from __future__ import annotations

import sys
from typing import Any

__all__ = ["MAC", "key_text", "mac_text"]

MAC = sys.platform == "darwin"

# Qt's own symbols on a Mac (qkeysequence.cpp): modifiers in the order Control, Option, Shift, Command, then the key
_MAC_MODIFIERS = (("Meta", "⌃"), ("Alt", "⌥"), ("Shift", "⇧"), ("Ctrl", "⌘"))
_MAC_KEYS = {
    "Esc": "\u238b", "Tab": "\u21e5", "Backtab": "\u21e4", "Backspace": "\u232b", "Return": "\u21b5", "Enter": "\u2324",
    "Del": "\u2326", "Clear": "\u2327", "Home": "\u2196", "End": "\u2198", "Left": "\u2190", "Up": "\u2191",
    "Right": "\u2192", "Down": "\u2193", "PgUp": "\u21de", "PgDown": "\u21df", "CapsLock": "\u21ea",
}


def _sequence(keys: Any) -> Any:
    from PySide6.QtGui import QKeySequence

    return keys if isinstance(keys, QKeySequence) else QKeySequence(keys)


def key_text(keys: Any) -> str:
    """A key sequence as the platform says it: ``keys`` in Qt's portable form ("Ctrl+K", "Ctrl+Shift+N"), a
    ``QKeySequence`` or a ``QKeySequence.StandardKey``. "Ctrl+K" on Linux and Windows, "⌘K" on a Mac; "" for no key
    (a standard key the platform does not bind)."""
    from PySide6.QtGui import QKeySequence

    sequence = _sequence(keys)
    if MAC and sys.platform != "darwin":  # a Mac's words asked for on another system (tests)
        return mac_text(sequence)
    return sequence.toString(QKeySequence.SequenceFormat.NativeText)


def mac_text(keys: Any) -> str:
    """A key sequence as Qt writes it on a Mac (⌃⌥⇧⌘ then the key, with no + between), on any system."""
    from PySide6.QtGui import QKeySequence

    portable = QKeySequence.SequenceFormat.PortableText
    chords = []
    for chord in _sequence(keys).toString(portable).split(", "):
        if not chord:
            continue
        # the key is what follows the last modifier: "Ctrl++" is Ctrl and +, "Ctrl+-" Ctrl and -
        parts, key = [], chord
        while True:
            head, plus, rest = key.partition("+")
            if plus and rest and head in dict(_MAC_MODIFIERS):
                parts.append(head)
                key = rest
            else:
                break
        chords.append("".join(sign for name, sign in _MAC_MODIFIERS if name in parts) + _MAC_KEYS.get(key, key))
    return ", ".join(chords)
