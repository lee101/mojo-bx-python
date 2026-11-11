"""Probe: what does parallelize() itself cost, and is threading alive in-process?

Two questions:
  1. Fixed overhead of a parallelize() launch with no work (per worker count).
  2. Whether repeated launches in one long-lived ctypes process keep all workers
     doing the work (a static first run would mask a pool that dies after use).

Usage: pixi run python bench/probe_overhead.py
"""

from __future__ import annotations

import ctypes
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
I = ctypes.c_int64
W = 4

SRC = r"""
from max.algorithm import parallelize


@export("probe_noop")
def probe_noop(threads: Int) abi("C") -> Int:
    comptime P = Pointer[Int, AnyOrigin[mut=True]]
    var counter = P(unsafe_from_address=2 ** 40)  # never dereferenced

    def work(k: Int) {imm}:
        pass

    for _ in range(1):
        parallelize(work, threads, threads)
    return threads


@export("probe_spin")
def probe_spin(bits: Int, words: Int, threads: Int) -> Int abi("C"):
    return 0
"""

NOOP_SRC = r"""
from max.algorithm import parallelize


@export("probe_noop")
def probe_noop(threads: Int) abi("C") -> Int:
    def work(k: Int) {imm}:
        pass
    parallelize(work, threads, threads)
    return threads
"""

FILL_SRC = r"""
from max.algorithm import parallelize

comptime P = Pointer[UInt64, AnyOrigin[mut=True]]


def _fill(p: P, count: Int, value: UInt64):
    var vector_end = count - count % 4
    var values = SIMD[DType.uint64, 4](value)
    for i in range(0, vector_end, 4):
        p.store(i, values)
    for i in range(vector_end, count):
        p.unsafe_store(i, value)


@export("probe_fill_parallel")
def probe_fill_parallel(bits: Int, words: Int, threads: Int) abi("C") -> Int:
    var b = P(unsafe_from_address=bits)
    var chunk = (words + threads - 1) // threads
    chunk = (chunk + 3) // 4 * 4

    def work(k: Int) {imm}:
        var lo = k * chunk
        var hi = min(lo + chunk, words)
        if lo < hi:
            _fill(b.unsafe_offset(lo), hi - lo, ~UInt64(0))
    parallelize(work, threads, threads)
    return 0


@export("probe_fill_serial")
def probe_fill_serial(bits: Int, words: Int) abi("C") -> Int:
    var b = P(unsafe_from_address=bits)
    _fill(b, words, ~UInt64(0))
    return 0
"""


def build(src: str, name: str) -> ctypes.CDLL:
    path = Path(tempfile.mkdtemp()) / name
    srcpath = path.with_suffix(".mojo")
    srcpath.write_text(src)
    subprocess.run(["bash", "-c", f"mojo build --emit shared-lib {srcpath} -o {path}.so"],
                   check=True, capture_output=True)
    return ctypes.CDLL(f"{path}.so")


def main():
    lib = build(NOOP_SRC + FILL_SRC, "probe_overhead.so")
    noop = lib.probe_noop
    noop.argtypes, noop.restype = [I], I

    print("## parallelize launch cost, empty worker, median of 200 launches (us)")
    print("| workers | median us | min us |")
    print("| ---: | ---: | ---: |")
    for t in (1, 2, 4, 6, 8, 16):
        samples = []
        for _ in range(200):
            t0 = time.perf_counter()
            noop(t)
            samples.append((time.perf_counter() - t0) * 1e6)
        print(f"| {t} | {statistics.median(samples):.1f} | {min(samples):.1f} |")

    print()
    print("## fill 1 GiB of packed words, serial vs parallel, repeated in one process")
    bits = 1 << 33
    words = bits // 64
    par = lib.probe_fill_parallel
    par.argtypes, par.restype = [I, I, I], I
    ser = lib.probe_fill_serial
    ser.argtypes, ser.restype = [I, I], I
    buf = np.zeros(words, dtype=np.uint64)
    addrp = int(buf.ctypes.data)
    print("| round | serial us | t1 | t2 | t4 | t6 | t8 |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for round_index in range(6):
        ser(addrp, words)
        t0 = time.perf_counter()
        ser(addrp, words)
        ser_time = (time.perf_counter() - t0) * 1e6
        cells = []
        for t in (1, 2, 4, 6, 8):
            buf[:] = 0
            t0 = time.perf_counter()
            par(addrp, words, t)
            cells.append(f"{(time.perf_counter() - t0) * 1e6:.0f}")
            assert (buf == np.uint64(0xFFFF_FFFF_FFFF_FFFF)).all(), f"round {round_index} t{t} incorrect"
        print(f"| {round_index} | {ser_time:.0f} | " + " | ".join(cells) + " |", flush=True)


if __name__ == "__main__":
    main()
