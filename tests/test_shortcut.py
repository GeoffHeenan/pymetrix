"""The desktop shortcut: on Linux a .desktop launcher whose command starts this Python's pymetrix app (run here with
--help); on Windows, with PowerShell replaced, the folders asked for, pythonw.exe chosen and the .lnk script sent;
pymetrix-gui --shortcut makes them without opening a window; --latest adds the flag to the shortcut, and the launcher's
update pulls a git checkout, saying what it did and never stopping the start."""

import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from pymetrix.ui import shortcut


def test_linux_launcher(tmp_path, monkeypatch):
    if sys.platform != "linux":
        pytest.skip("the .desktop launcher is Linux's")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))
    monkeypatch.setattr(shortcut, "_xdg_desktop", lambda: tmp_path / "Desktop")
    made = shortcut.create_shortcut(home=tmp_path)
    assert made == [tmp_path / "share" / "applications" / "pymetrix.desktop", tmp_path / "Desktop" / "pymetrix.desktop"]
    text = made[0].read_text(encoding="utf-8")
    assert "Name=PyMetrix" in text and "-m pymetrix_app" in text and "Icon=" in text and "pymetrix.png" in text and f"Path={tmp_path}" in text and "Terminal=false" in text
    command = shlex.split(next(line for line in text.splitlines() if line.startswith("Exec=")).removeprefix("Exec="))
    assert command[0] == sys.executable and command[1:3] == ["-m", "pymetrix_app"]
    done = subprocess.run([*command[:3], "--help"], capture_output=True, text=True, timeout=120)  # the command works
    assert done.returncode == 0 and "--shortcut" in done.stdout


def test_windows_link(tmp_path, monkeypatch):
    scripts = []

    def powershell(script):
        scripts.append(script)
        return f"{tmp_path / 'OneDrive' / 'Desktop'}\n{tmp_path / 'Programs'}" if "GetFolderPath" in script else ""

    monkeypatch.setattr(shortcut.sys, "platform", "win32")
    monkeypatch.setattr(shortcut, "_powershell", powershell)
    exe = tmp_path / "venv" / "Scripts" / "python.exe"
    exe.parent.mkdir(parents=True)
    exe.touch()
    (exe.parent / "pythonw.exe").touch()
    monkeypatch.setattr(shortcut.sys, "executable", str(exe))
    made = shortcut.create_shortcut(home=tmp_path / "it's here")
    assert made == [tmp_path / "OneDrive" / "Desktop" / "PyMetrix.lnk", tmp_path / "Programs" / "PyMetrix.lnk"]
    link = scripts[1]
    assert "WScript.Shell" in link and str(exe.parent / "pythonw.exe") in link and "-m pymetrix_app" in link
    assert "it''s here" in link  # PowerShell's single quotes doubled
    (exe.parent / "pythonw.exe").unlink()
    assert shortcut._gui_python() == exe  # no pythonw.exe: python.exe


def test_the_gui_command_makes_it_without_a_window(monkeypatch):
    made = []
    monkeypatch.setattr(shortcut, "create_shortcut", lambda latest=False: made.append(latest) or [Path("x")])
    from pymetrix.ui.qt import main

    main(["--shortcut"])
    main(["--shortcut", "--latest"])
    assert made == [False, True]


def test_latest_shortcut(tmp_path, monkeypatch):
    if sys.platform != "linux":
        pytest.skip("the .desktop launcher is Linux's")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))
    monkeypatch.setattr(shortcut, "_xdg_desktop", lambda: tmp_path / "Desktop")
    made = shortcut.create_shortcut(home=tmp_path, latest=True)
    command = shlex.split(next(line for line in made[0].read_text(encoding="utf-8").splitlines()
                               if line.startswith("Exec=")).removeprefix("Exec="))
    assert command[1:4] == ["-m", "pymetrix_app", "--latest"]


def test_update_pulls_a_checkout(tmp_path):
    from pymetrix_app import update

    def git(*args, cwd):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True,
                       capture_output=True)

    origin, clone = tmp_path / "origin", tmp_path / "clone"
    origin.mkdir()
    git("init", "-q", "-b", "main", cwd=origin)
    (origin / "a.txt").write_text("1")
    git("add", ".", cwd=origin)
    git("commit", "-qm", "one", cwd=origin)
    git("clone", "-q", str(origin), str(clone), cwd=tmp_path)
    assert update(clone)[1].startswith("Up to date at ")
    (origin / "a.txt").write_text("2")
    git("commit", "-qam", "two", cwd=origin)
    updated, line = update(clone)
    assert updated and line.startswith("Updated ") and (clone / "a.txt").read_text() == "2"
    (clone / "a.txt").write_text("local")  # a local change in the way of the next pull
    git("commit", "-qam", "local", cwd=clone)
    (origin / "a.txt").write_text("3")
    git("commit", "-qam", "three", cwd=origin)
    updated, line = update(clone)
    assert not updated and line.startswith("Could not update") and line.endswith("starting the build as it is")


def test_update_without_a_checkout(monkeypatch):
    import pymetrix_app

    monkeypatch.setattr(pymetrix_app, "checkout", lambda: None)
    assert pymetrix_app.update() == (False, "Installed from a package, not a git checkout: --latest has nothing to update")
