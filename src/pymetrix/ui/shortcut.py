"""A desktop shortcut for the app: ``pymetrix-gui --shortcut`` (or ``python -m pymetrix.ui.shortcut``).

The shortcut starts this Python with ``-m pymetrix_app``, so it opens the PyMetrix installed where the command was
run (a virtual environment's, if one is active) and needs no activation later. On Windows it is a ``.lnk`` on the
desktop and in the Start menu, run by ``pythonw.exe`` so no console window opens; the desktop is the one Windows reports
(a desktop moved to OneDrive included). On Linux it is a ``.desktop`` launcher in the applications menu and on the
desktop; on macOS a ``PyMetrix.command`` file on the desktop.

``pymetrix-gui --shortcut --latest`` makes a shortcut that starts the app with ``--latest``: for PyMetrix installed
from a git checkout (``pip install -e``), each start first pulls the branch's latest commits, so the icon always
runs the most recent build.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from pymetrix_app import ICON, ICON_WINDOWS

__all__ = ["create_shortcut"]

NAME = "PyMetrix"
COMMENT = "The PyMetrix econometrics workbench"
MODULE = "pymetrix_app"  # the launcher: its opening screen shows before pymetrix loads



def _icon_location(target: Path) -> str:
    """The shortcut's icon: PyMetrix's, else the interpreter's own."""
    return f"{ICON_WINDOWS},0" if ICON_WINDOWS.is_file() else f"{target},0"


def _gui_python() -> Path:
    """The interpreter to start the app with: pythonw.exe beside this python.exe on Windows (no console window)."""
    exe = Path(sys.executable)
    if sys.platform == "win32":
        quiet = exe.with_name("pythonw.exe")
        if quiet.exists():
            return quiet
    return exe


def _powershell(script: str) -> str:
    done = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                          capture_output=True, text=True, check=True)
    return done.stdout.strip()


def _windows_folders() -> list[Path]:
    """The desktop and the Start menu's Programs folder, as Windows reports them."""
    out = _powershell("[Environment]::GetFolderPath('Desktop'); [Environment]::GetFolderPath('Programs')")
    return [Path(line) for line in out.splitlines() if line.strip()]


def _windows_link(path: Path, target: Path, arguments: str, folder: Path) -> None:
    def quoted(text: object) -> str:
        return "'" + str(text).replace("'", "''") + "'"

    _powershell("$s = (New-Object -ComObject WScript.Shell).CreateShortcut(" + quoted(path) + "); "
                f"$s.TargetPath = {quoted(target)}; $s.Arguments = {quoted(arguments)}; "
                f"$s.WorkingDirectory = {quoted(folder)}; $s.Description = {quoted(COMMENT)}; "
                f"$s.IconLocation = {quoted(_icon_location(target))}; $s.Save()")


def _xdg_desktop() -> Path:
    try:
        found = subprocess.run(["xdg-user-dir", "DESKTOP"], capture_output=True, text=True, check=True).stdout.strip()
        if found:
            return Path(found)
    except (OSError, subprocess.CalledProcessError):
        pass
    return Path.home() / "Desktop"


def create_shortcut(home: Path | None = None, latest: bool = False) -> list[Path]:
    """Create the app's shortcuts for this platform and return their paths. ``home`` (the working folder the app starts
    in) defaults to the user's home folder; ``latest`` makes the app update its checkout each time it starts."""
    folder = Path(home) if home is not None else Path.home()
    python = _gui_python()
    module = f"{MODULE} --latest" if latest else MODULE
    made: list[Path] = []
    if sys.platform == "win32":
        for place in _windows_folders():
            place.mkdir(parents=True, exist_ok=True)
            link = place / f"{NAME}.lnk"
            _windows_link(link, python, f"-m {module}", folder)
            made.append(link)
        return made
    if sys.platform == "darwin":
        desktop = Path.home() / "Desktop"
        desktop.mkdir(parents=True, exist_ok=True)
        path = desktop / f"{NAME}.command"
        path.write_text(f'#!/bin/sh\ncd "{folder}"\nexec "{python}" -m {module} "$@"\n', encoding="utf-8")
        path.chmod(0o755)
        return [path]
    entry = (f"[Desktop Entry]\nType=Application\nName={NAME}\nComment={COMMENT}\n"
             f'Exec="{python}" -m {module} %f\nPath={folder}\nIcon={ICON}\nTerminal=false\nCategories=Science;Education;\n')
    data_home = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    for place in (data_home / "applications", _xdg_desktop()):
        place.mkdir(parents=True, exist_ok=True)
        path = place / "pymetrix.desktop"  # the launcher's id stays lower case
        path.write_text(entry, encoding="utf-8")
        path.chmod(0o755)
        made.append(path)
    return made


def main(latest: bool = False) -> None:
    for path in create_shortcut(latest=latest):
        print(f"made {path}")
    if latest:
        from pymetrix_app import checkout

        if checkout() is None:
            print("note: PyMetrix is installed from a package, not a git checkout, so --latest will not update it")


if __name__ == "__main__":
    main()
