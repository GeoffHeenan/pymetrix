"""Tabulate the null distribution of Quandt-Andrews' sup-Wald statistic: run once, offline.

    python scripts/make_quandt_andrews_table.py        # writes src/pymetrix/stats/quandt_andrews.npz

The limit (Andrews, 1993) is the sup over r in [trim, 1 - trim] of |B(r) - r B(1)|^2 / (r (1 - r)), B a
k-dimensional Brownian motion. It is simulated with REPS draws of a STEPS-step random walk for k = 1 to K_MAX
coefficients; each draw serves every trimming on TRIMS, so the trimmings share their draws. The table keeps the
quantiles at PROBS, dense in the upper tail where p-values and critical values are read. ``QuandtAndrews`` reads
it and simulates only for a k or a trimming the table does not hold. tests/test_breaks.py checks the table against
Andrews' published critical values.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPS, STEPS, SEED = 100_000, 5000, 19930701
K_MAX = 20
TRIMS = np.round(np.arange(0.05, 0.451, 0.05), 2)
PROBS = np.unique(np.round(np.concatenate([np.arange(0, 0.99, 0.005), np.arange(0.99, 0.99991, 0.0002)]), 6))
OUT = Path(__file__).resolve().parents[1] / "src" / "pymetrix" / "stats" / "quandt_andrews.npz"


def draws(k: int) -> np.ndarray:
    """REPS draws of the limit for k coefficients, one column per trimming in TRIMS."""
    rng = np.random.default_rng([SEED, k])
    r = np.arange(1, STEPS + 1) / STEPS
    windows = [(r >= t) & (r <= 1 - t) for t in TRIMS]
    inner = windows[0]  # the widest window holds every other
    rr = r[inner]
    out = np.empty((REPS, len(TRIMS)))
    chunk = max(1, 2_000_000 // (STEPS * k))
    for start in range(0, REPS, chunk):
        m = min(chunk, REPS - start)
        B = np.cumsum(rng.standard_normal((m, STEPS, k)), axis=1) / np.sqrt(STEPS)
        bridge = B[:, inner, :] - rr[None, :, None] * B[:, -1:, :]
        wald = np.sum(bridge**2, axis=2) / (rr * (1 - rr))[None, :]
        for j, w in enumerate(windows):
            out[start:start + m, j] = wald[:, w[inner]].max(axis=1)
    return out


def quantiles(k: int) -> np.ndarray:
    d = draws(k)
    return np.quantile(d, PROBS, axis=0).T  # trims x probs


def main() -> int:
    with ProcessPoolExecutor() as pool:
        table = np.stack(list(pool.map(quantiles, range(1, K_MAX + 1))))  # k x trims x probs
    np.savez_compressed(OUT, k=np.arange(1, K_MAX + 1), trim=TRIMS, probs=PROBS, quantiles=table.astype(np.float32),
                        reps=REPS, steps=STEPS, seed=SEED)
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
