"""Probe: two cheaper ways to thread the packed-word loops.

A) same-thread interleaving: block k executes on the thread that called it,
   so there is no worker-launch cost at all.  Trades away latency hiding.
B) parallelize(..., workers <= cap) with the grid already cut to the worker
   count, dropping the 3-6 ms fixed launch (wait/spawn/rejoin) to its floor.

Both are compared against serial and against the 4-worker grid on the same
buffer, alternating, for the four bitset kernels.

Usage: pixi run python bench/probe_interleave.py [bits] [trials]
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

I = ctypes.c_int64
FULL = 0xFFFF_FFFF_FFFF_FFFF

SRC = r"""
from std.bit import pop_count
from max.algorithm import parallelize

comptime W = 4
comptime P = Pointer[UInt64, AnyOrigin[mut=True]]


def _fill(p: P, count: Int, v: UInt64):
    var ve = count - count % W
    var vals = SIMD[DType.uint64, W](v)
    for i in range(0, ve, W):
        p.store(i, vals)
    for i in range(ve, count):
        p.unsafe_store(i, v)


def _binary(a: P, b: P, count: Int, op: Int):
    var ve = count - count % W
    for i in range(0, ve, W):
        var l = a.unsafe_load[width=W](i)
        var r = b.unsafe_load[width=W](i)
        if op == 0:
            a.store(i, l & r)
        elif op == 1:
            a.store(i, l | r)
        else:
            a.store(i, l ^ r)


def _popcount(p: P, count: Int) -> Int:
    var total = 0
    var ve = count - count % W
    for i in range(0, ve, W):
        total += Int(pop_count(p.unsafe_load[width=W](i)).reduce_add())
    for i in range(ve, count):
        total += Int(pop_count(p.unsafe_load(i)))
    return total


def _invert(p: P, count: Int):
    var ve = count - count % W
    for i in range(0, ve, W):
        p.store(i, ~p.unsafe_load[width=W](i))
    for i in range(ve, count):
        p.unsafe_store(i, ~p.unsafe_load(i))


def _chunk(words: Int, parts: Int) -> Int:
    var c = (words + parts - 1) // parts
    return (c + W - 1) // W * W


@export("probe_interleave_set")
def probe_interleave_set(bits: Int, start: Int, count: Int, value: Int, parts: Int) abi("C"):
    var b = P(unsafe_from_address=bits)
    var words = (start + count + 63) // 64
    var c = _chunk(words, parts)
    var v = ~UInt64(0) if value != 0 else UInt64(0)
    def work(k: Int) {imm}:
        var lo = k * c
        var hi = min(lo + c, words)
        if lo < hi:
            _fill(b.unsafe_offset(lo), hi - lo, v)
    parallelize(work, parts, 1)


@export("probe_launch_set")
def probe_launch_set(bits: Int, start: Int, count: Int, value: Int, threads: Int) abi("C"):
    var b = P(unsafe_from_address=bits)
    var words = (start + count + 63) // 64
    var c = _chunk(words, threads)
    var v = ~UInt64(0) if value != 0 else UInt64(0)
    def work(k: Int) {imm}:
        var lo = k * c
        var hi = min(lo + c, words)
        if lo < hi:
            _fill(b.unsafe_offset(lo), hi - lo, v)
    parallelize(work, threads, threads)


@export("probe_launch_binary")
def probe_launch_binary(bits: Int, other: Int, size: Int, operation: Int, threads: Int) abi("C"):
    var a = P(unsafe_from_address=bits)
    var b = P(unsafe_from_address=other)
    var words = (size + 63) // 64
    var c = _chunk(words, threads)
    def work(k: Int) {imm}:
        var lo = k * c
        var hi = min(lo + c, words)
        if lo < hi:
            _binary(a.unsafe_offset(lo), b.unsafe_offset(lo), hi - lo, operation)
    parallelize(work, threads, threads)


@export("probe_interleave_binary")
def probe_interleave_binary(bits: Int, other: Int, size: Int, operation: Int, parts: Int) abi("C"):
    var a = P(unsafe_from_address=bits)
    var b = P(unsafe_from_address=other)
    var words = (size + 63) // 64
    var c = _chunk(words, parts)
    def work(k: Int) {imm}:
        var lo = k * c
        var hi = min(lo + c, words)
        if lo < hi:
            _binary(a.unsafe_offset(lo), b.unsafe_offset(lo), hi - lo, operation)
    parallelize(work, parts, 1)


@export("probe_interleave_invert")
def probe_interleave_invert(bits: Int, size: Int, parts: Int) abi("C"):
    var b = P(unsafe_from_address=bits)
    var words = (size + 63) // 64
    var c = _chunk(words, parts)
    def work(k: Int) {imm}:
        var lo = k * c
        var hi = min(lo + c, words)
        if lo < hi:
            _invert(b.unsafe_offset(lo), hi - lo)
    parallelize(work, parts, 1)


@export("probe_launch_invert")
def probe_launch_invert(bits: Int, size: Int, threads: Int) abi("C"):
    var b = P(unsafe_from_address=bits)
    var words = (size + 63) // 64
    var c = _chunk(words, threads)
    def work(k: Int) {imm}:
        var lo = k * c
        var hi = min(lo + c, words)
        if lo < hi:
            _invert(b.unsafe_offset(lo), hi - lo)
    parallelize(work, threads, threads)
"""


def build() -> ctypes.CDLL:
    path = Path(tempfile.mkdtemp()) / "probe_interleave"
    src = path.with_suffix(".mojo")
    src.write_text(SRC)
    result = subprocess.run(["bash", "-c", f"mojo build --emit shared-lib {src} -o {path}.so"],
                            capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(result.stderr[-4000:])
    return ctypes.CDLL(f"{path}.so")


def main():
    bits = int(sys.argv[1]) if len(sys.argv) > 1 else (1 << 33)
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    lib = build()
    serial = ctypes.CDLL(Path(__file__).resolve().parents[1] / "dist" / "libmojo-bx-python.so")
    words = bits // 64
    a = np.zeros(words, dtype=np.uint64)
    b_ = np.ones(words, dtype=np.uint64)
    ap, bp = int(a.ctypes.data), int(b_.ctypes.data)

    for name in ("probe_interleave_set", "probe_launch_set"):
        getattr(lib, name).argtypes = [I] * 5
    for name in ("probe_interleave_binary", "probe_launch_binary"):
        getattr(lib, name).argtypes = [I] * 5
    for name in ("probe_interleave_invert", "probe_launch_invert"):
        getattr(lib, name).argtypes = [I] * 3
    serial.mbx_bits_set_range.argtypes = [I] * 4
    serial.mbx_bits_binary.argtypes = [I] * 4
    serial.mbx_bits_invert.argtypes = [I] * 2

    variants = [
        ("serial", lambda: serial.mbx_bits_set_range(ap, 0, bits, 1),
         lambda: serial.mbx_bits_binary(ap, bp, bits, 0),
         lambda: serial.mbx_bits_invert(ap, bits)),
        ("interleave p=2", lambda: lib.probe_interleave_set(ap, 0, bits, 1, 2),
         lambda: lib.probe_interleave_binary(ap, bp, bits, 0, 2),
         lambda: lib.probe_interleave_invert(ap, bits, 2)),
        ("interleave p=4", lambda: lib.probe_interleave_set(ap, 0, bits, 1, 4),
         lambda: lib.probe_interleave_binary(ap, bp, bits, 0, 4),
         lambda: lib.probe_interleave_invert(ap, bits, 4)),
        ("interleave p=8", lambda: lib.probe_interleave_set(ap, 0, bits, 1, 8),
         lambda: lib.probe_interleave_binary(ap, bp, bits, 0, 8),
         lambda: lib.probe_interleave_invert(ap, bits, 8)),
        ("launch t=2", lambda: lib.probe_launch_set(ap, 0, bits, 1, 2),
         lambda: lib.probe_launch_binary(ap, bp, bits, 0, 2),
         lambda: lib.probe_launch_invert(ap, bits, 2)),
        ("launch t=4", lambda: lib.probe_launch_set(ap, 0, bits, 1, 4),
         lambda: lib.probe_launch_binary(ap, bp, bits, 0, 4),
         lambda: lib.probe_launch_invert(ap, bits, 4)),
    ]

    print(f"Machine: {platform.node()} | {platform.platform()} | load {os.getloadavg()[0]:.1f}")
    print(f"{bits >> 30} Gi-bit = {bits >> 33} GiB per call, median ms of {trials} trials, "
          f"all results verified")
    print("| variant | set_range | iand | invert |")
    print("| --- | ---: | ---: | ---: |")
    results = {}
    for label, set_fn, bin_fn, inv_fn in variants:
        cells = []
        for fn, check in ((set_fn, lambda: (a == FULL).all()),
                          (bin_fn, lambda: (a == 0).all()),
                          (inv_fn, lambda: (a == FULL).all())):
            samples = []
            for _ in range(trials):
                a[:] = 0
                t0 = time.perf_counter()
                fn()
                samples.append((time.perf_counter() - t0) * 1e6)
                assert check(), f"{label} incorrect"
            cells.append(statistics.median(samples) / 1000)
        results[label] = cells
        print(f"| {label} | " + " | ".join(f"{c:.1f}" for c in cells) + " |", flush=True)

    base = results["serial"]
    print()
    print("speedup vs serial:")
    print("| variant | set_range | iand | invert |")
    print("| --- | ---: | ---: | ---: |")
    for label, cells in results.items():
        if label == "serial":
            continue
        print(f"| {label} | " + " | ".join(f"{s / c:.2f}x" for s, c in zip(base, cells)) + " |")


if __name__ == "__main__":
    main()
