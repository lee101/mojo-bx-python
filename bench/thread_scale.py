"""Kernel-level parallel-vs-serial A/B for the bitset word loops.

Builds src/capi.mojo into a shared library once, then times the packed-word
kernels at 1, 2, 4, 6 and 8 threads by calling the exported entry points with
an explicit worker count, alternating serial and parallel calls in one process.

Usage: pixi run python bench/thread_scale.py [reps]
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
I, F = ctypes.c_int64, ctypes.c_double


def build_probe() -> Path:
    """Build the experimental per-worker entry points into a scratch library."""
    out = Path(tempfile.mkdtemp()) / "probe_kernels.so"
    subprocess.run(["bash", "-c",
                    f"mojo build --emit shared-lib {ROOT / 'bench/probe_kernels.mojo'} -o {out}"],
                   check=True, capture_output=True)
    return out


def load():
    """Probe library for the parallel paths, production library for serial."""
    lib = ctypes.CDLL(build_probe())
    base = ctypes.CDLL(ROOT / "dist" / "libmojo-bx-python.so")
    sigs = {
        "mbx_bits_set_range": ([I] * 4, None),
        "mbx_bits_count_range": ([I] * 3, I),
        "mbx_bits_binary": ([I] * 4, None),
        "mbx_bits_invert": ([I] * 2, None),
        # experimental per-worker entry points
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


def words_for(bits):
    return (bits + 63) // 64


def timeit(call, reps):
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        call()
        samples.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(samples)


def best(call, reps, rounds=3):
    return min(timeit(call, reps) for _ in range(rounds))


CASES = []


def case(name, prepare):
    CASES.append((name, prepare))


def _set_range_len(bits):
    def call(lib, threads):
        a = np.zeros(words_for(bits), dtype=np.uint64)
        if threads == 0:
            lib.mbx_bits_set_range(a.ctypes.data, bits // 8, bits // 2, 1)
        else:
            lib.mbx_probe_set_range(a.ctypes.data, bits // 8, bits // 2, 1, threads)
    return call


def _count_len(bits):
    def call(lib, threads):
        a = np.zeros(words_for(bits), dtype=np.uint64)
        lo, hi = bits // 128, bits // 64
        a[lo:hi] = 0xFFFF_FFFF_FFFF_FFFF
        if threads == 0:
            r = lib.mbx_bits_count_range(a.ctypes.data, 0, bits)
        else:
            r = lib.mbx_probe_count_range(a.ctypes.data, 0, bits, threads)
        assert r == (hi - lo) * 64, (r, (hi - lo) * 64)
    return call


def _binary_len(bits):
    def call(lib, threads):
        a = np.zeros(words_for(bits), dtype=np.uint64)
        b = np.ones(words_for(bits), dtype=np.uint64)
        if threads == 0:
            lib.mbx_bits_binary(a.ctypes.data, b.ctypes.data, bits, 0)
        else:
            lib.mbx_probe_binary(a.ctypes.data, b.ctypes.data, bits, 0, threads)
    return call


def _invert_len(bits):
    def call(lib, threads):
        a = np.zeros(words_for(bits), dtype=np.uint64)
        if threads == 0:
            lib.mbx_bits_invert(a.ctypes.data, bits)
        else:
            lib.mbx_probe_invert(a.ctypes.data, bits, threads)
    return call


LENGTHS = [1 << 18, 1 << 20, 1 << 22, 1 << 24, 1 << 26, 1 << 28, 1 << 30]
THREADS = [1, 2, 4, 6, 8]

for n in LENGTHS:
    case(f"set_range {n >> 20}Mi-bit", _set_range_len(n))
    case(f"count_range {n >> 20}Mi-bit", _count_len(n))
    case(f"iand {n >> 20}Mi-bit", _binary_len(n))
    case(f"invert {n >> 20}Mi-bit", _invert_len(n))


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    lib = load()
    print(f"Machine: {platform.node()} | {platform.platform()} | load {os.getloadavg()[0]:.1f}")
    print("median us, best of 3 rounds; serial = production path, tN = N workers")
    header = "| case | serial | " + " | ".join(f"t{t}" for t in THREADS) + " | best speedup |"
    print(header)
    print("| --- | ---: | " + " | ".join("---:" for _ in THREADS) + " | ---: |")
    for name, call in CASES:
        serial = best(lambda: call(lib, 0), reps)
        par = {t: best(lambda t=t: call(lib, t), reps) for t in THREADS}
        speed = {t: serial / par[t] for t in THREADS}
        bt = max(speed, key=speed.get)
        cells = " | ".join(f"{par[t]:.1f} ({speed[t]:.2f}x)" for t in THREADS)
        print(f"| {name} | {serial:.1f} | {cells} | t{bt} {speed[bt]:.2f}x |", flush=True)


if __name__ == "__main__":
    main()
