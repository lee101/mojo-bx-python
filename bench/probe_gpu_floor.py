"""CPU kernel time next to the fixed GPU copy floor, so the gate is re-checkable.

The CPU half calls the real kernels at caller sizes.  The GPU half runs
bench/probe_gpu_floor.mojo, which copies the same volume up and back with no
kernel, synchronising each time: that is the irreducible cost any GPU path
would pay, and a kernel can only add to it.

Usage: pixi run python bench/probe_gpu_floor.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from bx._lib import addr, lib  # noqa: E402

L = lib()


def best(fn, n=9):
    fn()
    b = float("inf")
    for _ in range(n):
        t = time.perf_counter()
        fn()
        b = min(b, time.perf_counter() - t)
    return b


def cpu_score(ncols):
    from bx.align.score import _native_table, hox70

    a = np.frombuffer((b"ACGT-" * (ncols // 5 + 1))[:ncols], dtype=np.uint8)
    b = np.frombuffer((b"AGGT-" * (ncols // 5 + 1))[:ncols], dtype=np.uint8)
    table = _native_table(hox70)
    ms = best(lambda: L.mbx_score_texts(addr(a), addr(b), ncols, addr(table), table.shape[1],
                                        45, 45, 5.0, 1.0))
    return {"ncols": ncols, "bytes": ncols * 2, "cpu_raw_ms": round(ms * 1e3, 3)}


def main():
    print(json.dumps([cpu_score(1_000_000), cpu_score(100_000_000)], indent=2))
    r = subprocess.run(["pixi", "run", "mojo", "run", os.path.join(ROOT, "bench", "probe_gpu_floor.mojo")],
                       capture_output=True, text=True, cwd=ROOT)
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr.strip()[-800:])


if __name__ == "__main__":
    main()
