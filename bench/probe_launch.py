"""Probe: is there a cheaper per-worker launch primitive than parallelize()?

Compares the wall-clock cost of launching one trivial unit of work three ways:
  * a serial loop over N independent calls (what the current code does),
  * parallelize(func, N, workers)  (chunked by the runtime),
  * parallelize over 4x more work items than workers (runtime load-balances).

Usage: pixi run python bench/probe_launch.py
"""

from __future__ import annotations

import ctypes
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

I = ctypes.c_int64

SRC = r"""
from max.algorithm import parallelize

comptime P = Pointer[UInt64, AnyOrigin[mut=True]]


def _touch(p: P, lo: Int, hi: Int):
    for i in range(lo, hi):
        p.unsafe_store(i, p.unsafe_load(i) + UInt64(1))


@export("probe_serial")
def probe_serial(bits: Int, words: Int, workers: Int) abi("C") -> Int:
    var b = P(unsafe_from_address=bits)
    var chunk = (words + workers - 1) // workers
    for k in range(workers):
        var lo = k * chunk
        var hi = min(lo + chunk, words)
        if lo < hi:
            _touch(b, lo, hi)
    return 0


@export("probe_parallel")
def probe_parallel(bits: Int, words: Int, workers: Int) abi("C") -> Int:
    var b = P(unsafe_from_address=bits)
    var chunk = (words + workers - 1) // workers
    def work(k: Int) {imm}:
        var lo = k * chunk
        var hi = min(lo + chunk, words)
        if lo < hi:
            _touch(b, lo, hi)
    parallelize(work, workers, workers)
    return 0


@export("probe_parallel_items")
def probe_parallel_items(bits: Int, words: Int, workers: Int) abi("C") -> Int:
    var b = P(unsafe_from_address=bits)
    var items = workers * 4
    var chunk = (words + items - 1) // items
    def work(k: Int) {imm}:
        var lo = k * chunk
        var hi = min(lo + chunk, words)
        if lo < hi:
            _touch(b, lo, hi)
    parallelize(work, items, workers)
    return 0
"""


def build() -> ctypes.CDLL:
    path = Path(tempfile.mkdtemp()) / "probe_launch"
    src = path.with_suffix(".mojo")
    src.write_text(SRC)
    subprocess.run(["bash", "-c", f"mojo build --emit shared-lib {src} -o {path}.so"],
                   check=True, capture_output=True)
    lib = ctypes.CDLL(f"{path}.so")
    for name in ("probe_serial", "probe_parallel", "probe_parallel_items"):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = [I, I, I], I
    return lib


def main():
    import numpy as np

    lib = build()
    rounds = 15
    print("## cost of launching independent units of work (median us over 15 trials)")
    print("| units | serial loop | parallelize(workers=4) | workers=8 |")
    print("| ---: | ---: | ---: | ---: |")
    for units in (1, 2, 4, 8, 16, 32):
        buf = np.zeros(units + 8, dtype=np.uint64)
        a = int(buf.ctypes.data)
        words = units

        def timeit(fn, workers):
            samples = []
            for _ in range(rounds):
                buf[:] = 0
                t0 = time.perf_counter()
                fn(a, words, workers)
                samples.append((time.perf_counter() - t0) * 1e6)
            assert (buf[:units] == 1).all(), (fn.__name__, units, workers, buf[:units])
            return statistics.median(samples)

        print(f"| {units} | {timeit(lib.probe_serial, units):.1f} | "
              f"{timeit(lib.probe_parallel, 4):.1f} | {timeit(lib.probe_parallel, 8):.1f} |",
              flush=True)

    print()
    print("## same, but each unit writes 1 Mi words (real work, so launch cost is a residual)")
    words = 1 << 20
    buf = np.zeros(words, dtype=np.uint64)
    a = int(buf.ctypes.data)
    print("| workers | serial us | parallelize us | parallel 4x items |")
    print("| ---: | ---: | ---: | ---: |")

    def timeit_words(fn, workers):
        samples = []
        for _ in range(7):
            buf[:] = 0
            t0 = time.perf_counter()
            fn(a, words, workers)
            samples.append((time.perf_counter() - t0) * 1e6)
        assert (buf == 1).all()
        return min(samples)

    for workers in (1, 2, 4, 6, 8):
        print(f"| {workers} | {timeit_words(lib.probe_serial, workers):.0f} | "
              f"{timeit_words(lib.probe_parallel, workers):.0f} | "
              f"{timeit_words(lib.probe_parallel_items, workers):.0f} |", flush=True)


if __name__ == "__main__":
    main()
