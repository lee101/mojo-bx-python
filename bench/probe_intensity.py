"""Measure throughput of the covered kernels at caller sizes.

Streams are the bytes each kernel must read plus the bytes it must write at the
sizes bench/bench.py uses.  GPU floor numbers are in bench/probe_gpu_floor.mojo;
this file only reports what the CPU path actually achieves, so the GPU gate in
the README can be re-checked against live numbers rather than restated.

Usage: pixi run python bench/probe_intensity.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from bx._lib import addr, lib  # noqa: E402

L = lib()


def timed(fn, n=9):
    fn()
    best = float("inf")
    for _ in range(n):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def case_bits():
    bits = np.zeros((10_000_000 + 63) // 64, dtype=np.uint64)
    other = np.zeros_like(bits)
    return {
        "bitset set_range": (lambda: L.mbx_bits_set_range(addr(bits), 1_000_000, 8_000_000, 1), 2.5e6),
        "bitset count_range": (lambda: L.mbx_bits_count_range(addr(bits), 0, 10_000_000), 2.5e6),
        "bitset iand": (lambda: L.mbx_bits_binary(addr(bits), addr(other), 10_000_000, 0), 2.5e6),
    }


def case_align():
    from bx.align.score import _native_table, hox70

    a_s, b_s = "ACGT-" * 200_000, "AGGT-" * 200_000
    a = np.frombuffer(a_s.encode(), dtype=np.uint8)
    b = np.frombuffer(b_s.encode(), dtype=np.uint8)
    table = _native_table(hox70)
    from bx.align.score import score_texts

    return {
        "score_texts API": (lambda: score_texts(hox70, a_s, b_s), 2.0e6),
        "score_texts raw": (
            lambda: L.mbx_score_texts(addr(a), addr(b), len(a), addr(table), table.shape[1],
                                      ord("-"), ord("-"), 5.0, 1.0), 2.0e6),
    }


def case_intervals():
    starts = np.arange(20_000, dtype=np.int64) * 3
    ends = starts + 11
    max_ends = np.maximum.accumulate(ends)
    qs = np.arange(2_000, dtype=np.int64) * 23
    qe = qs + 500
    out = np.zeros(2_000, dtype=np.int64)
    return {
        "overlap_counts raw": (
            lambda: L.mbx_overlap_counts(addr(starts), addr(ends), addr(max_ends), len(starts),
                                         addr(qs), addr(qe), len(qs), addr(out)), 0.0),
    }


def main():
    cases = {}
    for build in (case_bits, case_align, case_intervals):
        cases.update(build())
    rows = []
    for name, (fn, stream) in cases.items():
        t = timed(fn)
        rows.append({"case": name, "ms": round(t * 1e3, 3),
                     "stream_MB": round(stream / 1e6, 3),
                     "GB_per_s": round(stream / t / 1e9, 2) if stream else None})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
