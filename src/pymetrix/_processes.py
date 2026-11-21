"""Long, independent pieces of work (Metropolis-Hastings chains, estimation starts) each in their own Python process.

Each job is a picklable function and its arguments. A process is started as ``python -m pymetrix._processes <file>``
with the job pickled on standard input, one linear-algebra thread, and this pymetrix on its path; the result is
pickled to the file. Not ``multiprocessing``: on Windows and macOS that imports the calling script again in each
process, which a script without ``if __name__ == "__main__":`` cannot survive (and exported pymetrix scripts have none).
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .errors import PyMetrixError

__all__ = ["run_in_processes"]

_THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


class ProcessError(PyMetrixError):
    """A job failed in its process."""


def run_in_processes(jobs: Sequence[tuple[Callable[..., Any], tuple]], workers: int,
                     done: Callable[[int, Any], Any] | None = None, what: str = "job") -> list[Any] | None:
    """The results of ``func(*args)`` for each job, ``workers`` processes at a time; ``done(i, result)`` as each ends.
    None if the jobs cannot be sent to another process (not picklable, or defined where another process cannot import
    it, such as the calling script): run them here then."""
    try:
        payloads = [pickle.dumps(job) for job in jobs]
    except Exception:  # noqa: BLE001 - whatever cannot be pickled
        return None
    env = dict(os.environ)
    for name in _THREAD_VARIABLES:
        env[name] = "1"
    package_root = str(Path(__file__).resolve().parents[1])  # so each process imports this pymetrix
    env["PYTHONPATH"] = os.pathsep.join([package_root, *[p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]])
    results: list[Any] = [None] * len(jobs)
    pending = list(range(len(jobs)))
    running: dict[int, subprocess.Popen] = {}
    logs: dict[int, Any] = {}  # each process's error output, in a file: a pipe could fill and stall it
    with tempfile.TemporaryDirectory(prefix="pymetrix-") as folder:
        def out(j: int) -> Path:
            return Path(folder, f"{what}{j}.pkl")

        def stop_all() -> None:
            for k, proc in running.items():
                proc.kill()
                proc.wait()
                logs[k].close()

        while pending or running:
            while pending and len(running) < max(1, workers):
                j = pending.pop(0)
                logs[j] = open(Path(folder, f"{what}{j}.err"), "wb")  # noqa: SIM115 - closed when the process ends
                proc = subprocess.Popen([sys.executable, "-m", "pymetrix._processes", str(out(j))], stdin=subprocess.PIPE,
                                        stdout=subprocess.DEVNULL, stderr=logs[j], env=env)
                assert proc.stdin is not None
                proc.stdin.write(payloads[j])
                proc.stdin.close()
                running[j] = proc
            for j, proc in list(running.items()):
                if proc.poll() is None:
                    continue
                del running[j]
                logs[j].close()
                error = Path(folder, f"{what}{j}.err").read_bytes().decode(errors="replace").strip()
                if proc.returncode != 0 or not out(j).exists():
                    stop_all()
                    raise ProcessError(f"{what} {j + 1} failed in its process: {error.splitlines()[-1] if error else proc.returncode}")
                status, value = pickle.loads(out(j).read_bytes())
                if status == "unsendable":  # e.g. a function defined in the calling script: the caller runs the jobs
                    stop_all()
                    return None
                if status != "ok":
                    stop_all()
                    raise ProcessError(f"{what} {j + 1}: {value}")
                results[j] = value
                if done is not None:
                    done(j, value)
            if running:
                time.sleep(0.05)
    return results


def main() -> None:
    try:
        func, args = pickle.loads(sys.stdin.buffer.read())
    except Exception as exc:  # noqa: BLE001 - defined where this process cannot import it
        result: tuple[str, Any] = ("unsendable", f"{type(exc).__name__}: {exc}")
    else:
        result = _run(func, args)
    with open(sys.argv[1], "wb") as f:
        pickle.dump(result, f)


def _run(func: Callable[..., Any], args: tuple) -> tuple[str, Any]:
    try:
        return ("ok", func(*args))
    except Exception as exc:  # noqa: BLE001 - reported by the calling process
        return ("error", f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
