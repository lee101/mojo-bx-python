"""Final measurement: production API before vs after the parallel pass.

Two modes:
  api  - public Python API (bx.bitset.BitSet) on the production library,
         serial vs threaded, at sizes a caller would actually pass.
  kernel - same buffers, raw ABI call, to separate Python/ctypes cost from the
         kernel cost.

The "before" library is built from git HEAD's src/capi.mojo into a scratch
library, so the comparison is against exactly the shipped serial code.

Usage: pixi run python bench/parallel_ab.py api [sizes_in_Mi_bit]
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

# The threshold is 2**24 words = 1024 Mi-bit of packed storage.  These sizes
# straddle it from well below to well above.
SIZES = [1, 8, 32, 128, 1024, 2048, 8192]


def build_before() -> Path:
    """Build the serial library from git HEAD's source."""
    src = subprocess.run(["git", "show", "HEAD:src/capi.mojo"], cwd=ROOT,
                         check=True, capture_output=True, text=True).stdout
    scratch = Path(tempfile.mkdtemp())
    (scratch / "capi.mojo").write_text(src)
    out = scratch / "libbefore.so"
    subprocess.run(["bash", "-c", f"mojo build --emit shared-lib {scratch / 'capi.mojo'} -o {out}"],
                   check=True, capture_output=True)
    return out


def median_ms(fn, reps, rounds=3):
    return min(statistics.median(_time(fn, reps)) for _ in range(rounds))


def _time(fn, reps):
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return samples


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "api"
    sys.path.insert(0, str(ROOT / "python"))
    from bx import bitset

    before_lib = ctypes.CDLL(build_before())
    after_lib = ctypes.CDLL(ROOT / "dist" / "libmojo-bx-python.so")
    for lib in (before_lib, after_lib):
        lib.mbx_bits_set_range.argtypes = [I] * 4
        lib.mbx_bits_count_range.argtypes = [I] * 3
        lib.mbx_bits_count_range.restype = I
        lib.mbx_bits_binary.argtypes = [I] * 4
        lib.mbx_bits_invert.argtypes = [I] * 2

    print(f"Machine: {platform.node()} | {platform.platform()} | load {os.getloadavg()[0]:.1f}")
    print(f"mode: {mode}")

    if mode == "api":
        # The Python module is bound to the production library.  For "before", swap
        # the module-level handle to the HEAD library.
        def swap(lib):
            bitset._NATIVE = lib

        print("| size | op | before (serial) | after | change |")
        print("| ---: | --- | ---: | ---: | ---: |")
        for mibit in SIZES:
            n = mibit << 20
            for op in ("set+count", "iand", "invert"):
                def run_once():
                    b = bitset.BitSet(n)
                    if op == "set+count":
                        b.set_range(n // 10, n // 2)
                        b.count_range(0, n)
                    elif op == "iand":
                        o = bitset.BitSet(n)
                        o.set_range(0, n // 3)
                        b.set_range(0, n // 2)
                        b.iand(o)
                    else:
                        b.set_range(0, n // 2)
                        b.invert()
                    return b

                swap(before_lib)
                t_before = median_ms(run_once, reps=5)
                swap(after_lib)
                t_after = median_ms(run_once, reps=5)
                change = t_before / t_after
                label = f"{change:.2f}x faster" if change >= 1 else f"{1 / change:.2f}x slower"
                print(f"| {mibit} Mi-bit | {op} | {t_before:.3f} ms | {t_after:.3f} ms | {label} |",
                      flush=True)
    else:
        # Kernel mode: identical starting content for both variants, so count
        # and invert cannot be timing a different amount of work.  The reset is
        # outside the clock and outside the median (it is done per repetition).
        PATTERN = np.uint64(0x0123_4567_89AB_CDEF)
        print("| size | op | before (serial) | after | change |")
        print("| ---: | --- | ---: | ---: | ---: |")
        for mibit in SIZES:
            n = mibit << 20
            words = (n + 63) // 64
            a = np.zeros(words, dtype=np.uint64)
            o = np.ones(words, dtype=np.uint64)
            ap, op_ = int(a.ctypes.data), int(o.ctypes.data)

            def call_only(lib, op):
                if op == "set_range":
                    lib.mbx_bits_set_range(ap, n // 10, n // 2, 1)
                elif op == "count_range":
                    lib.mbx_bits_count_range(ap, 0, n)
                elif op == "iand":
                    lib.mbx_bits_binary(ap, op_, n, 0)
                else:
                    lib.mbx_bits_invert(ap, n)

            def timed(lib, op):
                a[:] = PATTERN
                t0 = time.perf_counter()
                call_only(lib, op)
                return (time.perf_counter() - t0) * 1e3

            for op in ("set_range", "count_range", "iand", "invert"):
                t_before = median_ms(lambda: timed(before_lib, op), reps=7)
                t_after = median_ms(lambda: timed(after_lib, op), reps=7)
                change = t_before / t_after
                label = f"{change:.2f}x faster" if change >= 1 else f"{1 / change:.2f}x slower"
                print(f"| {mibit} Mi-bit | {op} | {t_before:.3f} ms | {t_after:.3f} ms | {label} |",
                      flush=True)


if __name__ == "__main__":
    main()
