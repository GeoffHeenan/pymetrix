"""The C/C++ stacks in a Windows CI job's crash dumps: the "Native stacks (Windows)" steps of tests.yml run this.

For each dump (the first three), cdb from the runner's Windows SDK debuggers is run three times, so that one part's
length or slowness cannot hide another's:

1. the exception and the faulting thread's stack (``.ecxr``, ``.exr -1``, ``kn 80``), printed in full, first, then
   the memory the first argument registers point to (``dps``: an object's first words, its vtable named if intact);
2. every thread's top frames (``~*kn 8``), summarised: how many threads, how many by the first module that is not
   Windows' own (a pile of one library's threads shows at a glance), and each distinct stack once, with its count;
3. ``!analyze -v``, with a time limit of its own and its output capped.

Symbols come from Microsoft's symbol server and Python's own; Qt's and PySide6's frames are named from their DLLs'
exports. The dumps themselves are kept as an artifact for a closer look.

    python .github/scripts/native_stacks.py [DUMP_FOLDER]     (default: $RUNNER_TEMP/crash-dumps)
"""

from __future__ import annotations

import collections
import os
import re
import subprocess
import sys
from pathlib import Path

# modules every Windows thread passes through: a thread is known by the first module on its stack that is not one
SYSTEM = {m.lower() for m in (
    "ntdll", "KERNELBASE", "kernel32", "win32u", "user32", "gdi32", "combase", "ucrtbase", "msvcrt", "msvcp_win",
    "mswsock", "ws2_32", "rpcrt4", "sechost", "advapi32", "ole32", "shell32", "vcruntime140", "vcruntime140_1")}
THREAD = re.compile(r"^\s*[.#]?\s*(\d+)\s+Id:\s*([0-9a-f]+\.[0-9a-f]+)", re.IGNORECASE)
FRAME = re.compile(r"^([0-9a-f]{2,})\s+[0-9a-f`]+\s+[0-9a-f`]+\s+(.+?)\s*$", re.IGNORECASE)
NOISE = re.compile(r"NatVis script (un)?loaded")
DISTINCT_STACKS = 40  # each distinct stack printed once, most common first, at most this many
ANALYZE_LINES = 300


def find_cdb() -> Path | None:
    for root in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles")):
        if root and (path := Path(root, "Windows Kits", "10", "Debuggers", "x64", "cdb.exe")).is_file():
            return path
    return None


def cdb(exe: Path, dump: Path, commands: str, timeout: float) -> list[str]:
    """cdb's output for ``commands`` on ``dump``, without its NatVis chatter; what came before a timeout, if one."""
    try:
        done = subprocess.run([str(exe), "-z", str(dump), "-c", f"{commands}; q"], capture_output=True, timeout=timeout)
        out, note = done.stdout, []
    except subprocess.TimeoutExpired as exc:
        out, note = exc.stdout or b"", [f"(cdb stopped after {timeout:.0f} s: the output above is what it gave)"]
    text = out.decode("utf-8", errors="replace")
    return [line for line in text.splitlines() if not NOISE.search(line)] + note


def threads(lines: list[str]) -> list[tuple[str, list[str]]]:
    """Each thread in ``~*kn`` output: its header line and its frames' call sites (module!function+offset)."""
    found: list[tuple[str, list[str]]] = []
    for line in lines:
        if THREAD.match(line):
            found.append((line.strip(), []))
        elif found and (frame := FRAME.match(line)):
            found[-1][1].append(frame.group(2))
    return found


def module(site: str) -> str:
    return re.split(r"[!+]", site, maxsplit=1)[0]


def top_module(frames: list[str]) -> str:
    """The first module on the stack that is not Windows' own; the innermost one if all are."""
    mods = [module(f) for f in frames]
    return next((m for m in mods if m.lower() not in SYSTEM), mods[0] if mods else "(no frames)")


def summary(lines: list[str]) -> list[str]:
    found = threads(lines)
    if not found:  # not the output expected: show it as it came, capped
        return ["(no threads found in cdb's output; its first 200 lines follow)", *lines[:200]]
    out = [f"{len(found)} threads. By the first module on each stack that is not Windows' own:"]
    by_module = collections.Counter(top_module(frames) for _, frames in found)
    out += [f"  {count:5d}  {name}" for name, count in by_module.most_common()]
    stacks = collections.defaultdict(list)  # the same frames, without addresses: one stack, however many threads
    for header, frames in found:
        stacks[tuple(re.sub(r"\+0x[0-9a-f]+$", "", f) for f in frames)].append(header)
    ordered = sorted(stacks.items(), key=lambda kv: -len(kv[1]))
    out.append(f"{len(stacks)} distinct stacks (top {DISTINCT_STACKS}, most common first; frames without offsets):")
    for frames, headers in ordered[:DISTINCT_STACKS]:
        ids = ", ".join(h.split("Id:")[0].strip(" .#") for h in headers[:12]) + (", ..." if len(headers) > 12 else "")
        out.append(f"--- {len(headers)} thread(s): {ids}")
        out += [f"      {f}" for f in frames]
    if len(ordered) > DISTINCT_STACKS:
        out.append(f"... and {len(ordered) - DISTINCT_STACKS} more distinct stacks, of "
                   f"{sum(len(h) for _, h in ordered[DISTINCT_STACKS:])} threads (the dump has them all)")
    return out


def main() -> int:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else Path(os.environ.get("RUNNER_TEMP", "."), "crash-dumps"))
    dumps = sorted(folder.glob("*.dmp")) if folder.is_dir() else []
    print(f"crash dumps: {len(dumps)}")
    if not dumps:
        return 0
    exe = find_cdb()
    if exe is None:
        print("cdb not found; the dumps are in this job's crash-dumps artifact")
        return 0
    runner_temp = os.environ.get("RUNNER_TEMP", ".")
    os.environ["_NT_SYMBOL_PATH"] = (f"srv*{runner_temp}\\symbols*https://msdl.microsoft.com/download/symbols;"
                                     f"{os.environ.get('pythonLocation', '')}")
    for dump in dumps[:3]:
        print(f"===== {dump.name}: the exception and the faulting thread", flush=True)
        print("\n".join(cdb(exe, dump, ".ecxr; .echo ----- the exception; .exr -1; "
                                       ".echo ----- the faulting thread; kn 80; "
                                       ".echo ----- memory at rcx and rdx (a call's first two arguments: for a member "
                                       "function, its object) and rdi; dps @rcx L8; dps @rdx L4; dps @rdi L8",
                                  timeout=240)), flush=True)
        print(f"===== {dump.name}: every thread", flush=True)
        print("\n".join(summary(cdb(exe, dump, "~*kn 8", timeout=180))), flush=True)
        print(f"===== {dump.name}: !analyze -v (first {ANALYZE_LINES} lines)", flush=True)
        print("\n".join(cdb(exe, dump, "!analyze -v", timeout=150)[:ANALYZE_LINES]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
