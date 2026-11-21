"""JDemetra+ 3, Eurostat's seasonal-adjustment software, run in Java through JPype: TRAMO-SEATS (``pymetrix_seasonal.TramoSeats``).

    pymetrix_seasonal.jdemetra.install()          # downloads the JDemetra+ libraries (about 7 MB) into ~/.pymetrix/jdemetra
    data.seasonal_adjust("y", TramoSeats())

It needs Java 21 or later (JDemetra+ 3.9 is built for it), found by JPype from ``JAVA_HOME`` or the system's default
Java, and JPype itself (``pip install "pymetrix-seasonal[jdemetra]"``). The libraries are JDemetra+'s own jars from Maven Central
(the EUPL 1.2 licence; not shipped with pymetrix): ``install()`` fetches them and checks each against the SHA-1 recorded
here, or ``JDPLUS_PATH`` names a folder that holds them already. The Java virtual machine starts once per Python
session, on first use, with these jars; a session whose JVM another package started without them cannot run
TRAMO-SEATS.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Any

from pymetrix.errors import SpecificationError

__all__ = ["VERSION", "install", "jar_folder", "shutdown"]

VERSION = "3.9.0"
_MAVEN = "https://repo1.maven.org/maven2"
_TRIES = 4  # each download, when Maven Central is busy
_JD = "eu/europa/ec/joinup/sat"

# each jar TRAMO-SEATS needs (jdplus-tramoseats-base-r and what Maven resolves under it): its path and SHA-1
JARS: dict[str, tuple[str, str]] = {
    **{f"jdplus-{name}-{VERSION}.jar": (f"{_JD}/jdplus-{name}/{VERSION}", sha1) for name, sha1 in (
        ("sa-base-api", "05308017d26f70dc5d785c4dff875530e2ab6c62"),
        ("sa-base-core", "5bcd03fa1b98dfc8ff0c0240409ce7c115790d09"),
        ("sa-base-protobuf", "e8cd61fad73bcd77ea45c56cffa130fd10b628bd"),
        ("toolkit-base-api", "4a66ed19da0829951b48333fac2a27b3978ec0b7"),
        ("toolkit-base-core", "6588cd5b0bf142e267428644518fa3286211ad89"),
        ("toolkit-base-protobuf", "72fc1b36bf767215e1b313bba4a4f3a6592d9cb9"),
        ("toolkit-base-r", "390ce6626203cec072dd715f3cf13d99bb4ac85c"),
        ("tramoseats-base-api", "5f816dcb9a6203c68492bc913a590234c020702e"),
        ("tramoseats-base-core", "e5305814390732f82c426923d17264cab8e3317c"),
        ("tramoseats-base-protobuf", "f5ee95dc260cb14932eba70092a7a59b553af5c3"),
        ("tramoseats-base-r", "a386a16b58b09467da66069831a0cc8f6a2a3a38"),
    )},
    "protobuf-java-4.36.1.jar": ("com/google/protobuf/protobuf-java/4.36.1", "b5db3504aebb79f4af24eb99246eabedb3859c1b"),
}

_NEEDS = ("pip install \"pymetrix-seasonal[jdemetra]\" for JPype; Java 21 or later (e.g. Eclipse Temurin 21), found from JAVA_HOME; "
          "and the JDemetra+ libraries: pymetrix_seasonal.jdemetra.install(), or JDPLUS_PATH set to a folder holding them")


def _home() -> Path:
    return Path(os.environ.get("PYMETRIX_HOME") or Path.home() / ".pymetrix") / "jdemetra" / VERSION


def _complete(folder: Path) -> bool:
    return all((folder / jar).is_file() for jar in JARS)


def jar_folder(path: str | None = None) -> str | None:
    """The folder with the JDemetra+ jars: ``path``, ``JDPLUS_PATH``, or where ``install()`` puts them; None when none
    of them holds every jar."""
    for folder in (path, os.environ.get("JDPLUS_PATH"), str(_home())):
        if folder and _complete(Path(folder)):
            return str(folder)
    return None


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def install(folder: str | None = None) -> str:
    """Download the JDemetra+ jars from Maven Central into ``folder`` (default ``~/.pymetrix/jdemetra/<version>``), each
    checked against its SHA-1; jars already there and intact are kept. Returns the folder."""
    import urllib.error
    import urllib.request

    target = Path(folder) if folder else _home()
    target.mkdir(parents=True, exist_ok=True)
    for jar, (where, sha1) in JARS.items():
        path = target / jar
        if path.is_file() and _sha1(path) == sha1:
            continue
        partial = path.with_suffix(".part")
        for attempt in range(_TRIES):
            try:
                with urllib.request.urlopen(f"{_MAVEN}/{where}/{jar}", timeout=120) as response:
                    partial.write_bytes(response.read())
                break
            except urllib.error.HTTPError as exc:  # Maven Central turns away requests that come too fast (429)
                if exc.code not in (429, 500, 502, 503, 504) or attempt == _TRIES - 1:
                    raise SpecificationError(f"{jar} could not be fetched from Maven Central ({exc}): try again later") from None
                time.sleep(5 * 2 ** attempt)
        if _sha1(partial) != sha1:
            partial.unlink()
            raise SpecificationError(f"{jar} from Maven Central does not match its recorded SHA-1: not installed")
        partial.replace(path)
    return str(target)


def shutdown() -> None:
    """Stop the Java virtual machine, if TRAMO-SEATS started one; nothing otherwise (JPype is not imported for it).

    The desktop app calls this as it quits, before Qt's windows are destroyed: on Windows, Java still running while Qt
    was torn down crashed the process. Java cannot start again in the same Python session afterwards."""
    jpype = sys.modules.get("jpype")
    if jpype is not None and jpype.isJVMStarted():
        jpype.shutdownJVM()


def _jvm_options() -> list[str]:
    """Options for the Java virtual machine: on Linux, a thread stack size (``-Xss``) as large as the main thread's.

    On Linux, Java (HotSpot) started from a process's main thread, as JPype starts it, takes that thread's stack to be
    no larger than ``-Xss`` (1 MB by default) and puts its guard pages there, an eighth of the usual 8 MB: Python code
    that ran fine before Java started could then overflow the stack (Python calls through C, such as Qt's events and
    widgets calling back into Python, go about 1,000 deep rather than 11,000). With Java's signal handler in place such
    an overflow ends the process with no report at all, not even Python's fatal-error traceback (as a test process on CI
    ended, in a run whose TRAMO-SEATS tests start Java). So ``-Xss`` is the main thread's own limit (``ulimit -s``;
    8 MB where it is unlimited), and Java leaves the main thread its whole stack. Java's own threads reserve that much
    address space each, committed only as they use it. Nothing elsewhere: other systems do not cut the main thread."""
    if not sys.platform.startswith("linux"):
        return []
    import resource

    soft, _ = resource.getrlimit(resource.RLIMIT_STACK)
    size = 8 * 1024 * 1024 if soft == resource.RLIM_INFINITY or soft <= 0 else soft
    return [f"-Xss{max(size, 1024 * 1024) // 1024}k"]


def classes(path: str | None = None) -> Any:
    """Start the JVM with the jars (once) and return ``jpype.JClass``, for looking up JDemetra+'s classes."""
    try:
        import jpype
    except ImportError:
        raise SpecificationError(f"TRAMO-SEATS runs JDemetra+ in Java, through JPype, which is not installed: {_NEEDS}") from None
    folder = jar_folder(path)
    if folder is None:
        raise SpecificationError(f"TRAMO-SEATS needs the JDemetra+ {VERSION} libraries: {_NEEDS}")
    if not jpype.isJVMStarted():
        try:
            jvm = jpype.getDefaultJVMPath()
        except Exception:  # noqa: BLE001 - JPype's JVMNotFoundException, among others
            raise SpecificationError(f"TRAMO-SEATS needs Java 21 or later, and none was found: {_NEEDS}") from None
        jpype.startJVM(jvm, *_jvm_options(), classpath=[str(Path(folder) / jar) for jar in JARS], convertStrings=False)
    try:
        jpype.JClass("jdplus.tramoseats.base.r.TramoSeats")
    except Exception as exc:  # noqa: BLE001 - UnsupportedClassVersionError (old Java) or a JVM started without the jars
        old = "UnsupportedClassVersion" in type(exc).__name__ or "class file version" in str(exc)
        raise SpecificationError(
            f"JDemetra+ {VERSION} needs Java 21 or later; this Java is older: {_NEEDS}" if old else
            "Java was started in this session without the JDemetra+ libraries (by another package?): "
            "start a new Python session and run TRAMO-SEATS before it") from None
    return jpype.JClass
