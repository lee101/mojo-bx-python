"""Probe: popcount read throughput, serial vs parallel, at large sizes.

count_range had no gain above the caller-visible range; this isolates whether
that is the counting read itself or the scratch/reduction around it.

Usage: pixi run python bench/probe_count_read.py [bits] [trials]
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

I = ctypes.c_int64

SRC = r"""
from std.bit import pop_count
from max.algorithm import parallelize

comptime W = 4
comptime P = Pointer[UInt64, AnyOrigin[mut=True]]


def _pop(p: P, n: Int) -> Int:
    var total = 0
    var ve = n - n % W
    for i in range(0, ve, W):
        total += Int(pop_count(p.unsafe_load[width=W](i)).reduce_add())
    for i in range(ve, n):
        total += Int(pop_count(p.unsafe_load(i)))
    return total


@export("probe_count_parallel")
def probe_count_parallel(bits: Int, words: Int, threads: Int, items: Int, scratch: Int) abi("C") -> Int:
    var b = P(unsafe_from_address=bits)
    var partial = Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=scratch)
    var chunk = (words + items - 1) // items
    def work(k: Int) {imm}:
        var lo = k * chunk
        var hi = min(lo + chunk, words)
        partial.unsafe_store(k, _pop(b.unsafe_offset(lo), hi - lo) if lo < hi else 0)
    parallelize(work, items, threads)
    var total = 0
    for k in range(items):
        total += Int(partial.unsafe_load(k))
    return total


@export("probe_count_serial")
def probe_count_serial(bits: Int, words: Int) abi("C") -> Int:
    return _pop(P(unsafe_from_address=bits), words)
"""


def main():
    bits = int(sys.argv[1]) if len(sys.argv) > 1 else (1 << 35)
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    scratch = Path(tempfile.mkdtemp())
    src = scratch / "p.mojo"
    src.write_text(SRC)
    subprocess.run(["bash", "-c", f"mojo build --emit shared-lib {src} -o {scratch / 'p.so'}"],
                   check=True, capture_output=True)
    lib = ctypes.CDLL(str(scratch / "p.so"))
    lib.probe_count_parallel.argtypes = [I] * 5
    lib.probe_count_parallel.restype = I
    lib.probe_count_serial.argtypes = [I] * 2
    lib.probe_count_serial.restype = I

    words = bits // 64
    # buffer + 16 int64 slots for partials, so the probe never allocates
    buf = np.zeros(words + 16, dtype=np.uint64)
    buf[:words] = 0xAAAA_AAAA_AAAA_AAAA
    base = int(buf.ctypes.data)
    partials = int(buf[words:].ctypes.data)
    expected = int(np.bitwise_count(buf[:words]).sum())

    def timed(fn):
        samples = []
        for _ in range(trials):
            t0 = time.perf_counter()
            got = fn()
            samples.append((time.perf_counter() - t0) * 1e6)
            assert got == expected, (got, expected)
        samples.sort()
        return samples[len(samples) // 2]

    print(f"{bits >> 30} Gi-bit = {bits >> 33} GiB read, median ms of {trials}, verified")
    serial = timed(lambda: lib.probe_count_serial(base, words))
    print(f"serial popcount: {serial / 1000:.1f} ms  ({bits / serial / 1e3:.1f} Gbit/s)")
    print("| grid | workers | ms | speedup |")
    print("| ---: | ---: | ---: | ---: |")
    for items in (4, 8, 16, 32):
        for threads in (2, 4):
            if threads > min(items, 4):
                continue
            buf[words:] = 0
            got = lib.probe_count_parallel(base, words, threads, items, partials)
            parts = buf[words:].copy()
            if got != expected:
                print(f"| {items} | {threads} | MISMATCH got {got} want {expected} "
                      f"partials {parts.tolist()} | - |")
                print("serial check:", lib.probe_count_serial(base, words))
                continue
            t = timed(lambda: lib.probe_count_parallel(base, words, threads, items, partials))
            print(f"| {items} | {threads} | {t / 1000:.1f} | {serial / t:.2f}x |", flush=True)


if __name__ == "__main__":
    main()
