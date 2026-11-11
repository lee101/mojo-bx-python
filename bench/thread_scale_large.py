"""Large-size thread-scaling trial for the packed-word bitset kernels.

This is the decisive experiment for the size threshold.  These kernels are
memory-bandwidth bound, so the crossover has to be located where the working
set is far beyond cache and TLB reach: this runs at 8 GiB and 32 GiB of packed
words per call (640 Mi and 2.56 Gi words), on buffers touched only by one
kernel, alternating serial and parallel calls in one process.

Per-worker entry points come from bench/probe_kernels.mojo (built to a scratch
library); serial baselines come from the production dist/ library.  Every call
result is verified before its timing is kept, so a fast-but-wrong split cannot
win here.

Usage: pixi run python bench/thread_scale_large.py [bits] [trials]
"""

from __future__ import annotations

import ctypes
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
I = ctypes.c_int64
FULL = 0xFFFF_FFFF_FFFF_FFFF


def build_probe() -> Path:
    out = Path(tempfile.mkdtemp()) / "probe_kernels.so"
    subprocess.run(["bash", "-c",
                    f"mojo build --emit shared-lib {ROOT / 'bench/probe_kernels.mojo'} -o {out}"],
                   check=True, capture_output=True)
    return out


def load():
    lib = ctypes.CDLL(build_probe())
    base = ctypes.CDLL(ROOT / "dist" / "libmojo-bx-python.so")
    sigs = {
        "mbx_bits_set_range": ([I] * 4, None),
        "mbx_bits_count_range": ([I] * 3, I),
        "mbx_bits_binary": ([I] * 4, None),
        "mbx_bits_invert": ([I] * 2, None),
        "mbx_probe_set_range": ([I] * 5, None),
        "mbx_probe_count_range": ([I] * 4, I),
        "mbx_probe_binary": ([I] * 5, None),
        "mbx_probe_invert": ([I] * 3, None),
    }
    for name, (args, restype) in sigs.items():
        fn = getattr(lib if name.startswith("mbx_probe") else base, name)
        fn.argtypes = args
        fn.restype = restype
        setattr(lib, name, fn)
    return lib


def addr(a: np.ndarray) -> int:
    return int(a.ctypes.data)


KERNELS = ("set_range", "count_range", "iand", "invert")


def run_case(lib, kernel, bits, threads, trials):
    """Median ms per call for one kernel and size, for each worker count in ``threads``."""
    words = (bits + 63) // 64
    a = np.zeros(words, dtype=np.uint64)
    b = np.ones(words, dtype=np.uint64) if kernel == "iand" else None
    # count_range: mark one contiguous run of words so the expected popcount is exact
    lo, hi = words // 4, words // 2
    expect_count = (hi - lo) * 64

    def call(t):
        if kernel == "set_range":
            if t == 0:
                lib.mbx_bits_set_range(addr(a), 0, bits, 1)
            else:
                lib.mbx_probe_set_range(addr(a), 0, bits, 1, t)
            assert (a == FULL).all(), "set_range mismatch"
        elif kernel == "count_range":
            a[:] = 0
            a[lo:hi] = FULL
            got = (lib.mbx_bits_count_range(addr(a), 0, bits) if t == 0
                   else lib.mbx_probe_count_range(addr(a), 0, bits, t))
            assert got == expect_count, (t, got, expect_count)
        elif kernel == "iand":
            if t == 0:
                lib.mbx_bits_binary(addr(a), addr(b), bits, 0)
            else:
                lib.mbx_probe_binary(addr(a), addr(b), bits, 0, t)
            assert (a == 0).all(), "iand mismatch"
        else:
            if t == 0:
                lib.mbx_bits_invert(addr(a), bits)
            else:
                lib.mbx_probe_invert(addr(a), bits, t)
            assert (a == FULL).all(), "invert mismatch"

    timings = {t: [] for t in threads}
    for _ in range(trials):
        for t in threads:
            a[:] = 0
            t0 = time.perf_counter()
            call(t)
            timings[t].append((time.perf_counter() - t0) * 1e6)
    return {t: statistics.median(v) for t, v in timings.items()}


def main():
    bits = int(sys.argv[1]) if len(sys.argv) > 1 else (1 << 35)
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    lib = load()
    print(f"Machine: {platform.node()} | {platform.platform()} | load {os.getloadavg()[0]:.1f}")
    print(f"{bits >> 30} Gi-bit packed = {bits >> 33} GiB per kernel call, "
          f"{(bits // 64) >> 20} Mi words, median ms of {trials} trials")
    threads = [0, 1, 2, 4, 6, 8]
    print("| kernel | serial | " + " | ".join(f"t{t}" for t in threads[1:]) + " |")
    print("| --- | ---: | " + " | ".join("---:" for _ in threads[1:]) + " |")
    for kernel in KERNELS:
        r = run_case(lib, kernel, bits, threads, trials)
        serial = r[0]
        cells = " | ".join(f"{r[t] / 1000:.1f} ({serial / r[t]:.2f}x)" for t in threads[1:])
        print(f"| {kernel} | {serial / 1000:.1f} ms | {cells} |", flush=True)


if __name__ == "__main__":
    main()
