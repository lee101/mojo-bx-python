"""Serial vs parallel equivalence across the size threshold.

Builds the *shipped* library once with the threshold forced to 0 (every packed-
word loop threads) and once with it at the real 2**24 words (every size below
that runs serial), then calls the exported entry points directly on identical
buffers and compares bit-for-bit.

The in-tree pytest suite cannot cover this: its sizes stay below the shipped
threshold, so it never enters the parallel branch.  This is the test that does.

Exit status is nonzero on any mismatch.  Usage: pixi run python bench/parallel_equivalence.py
"""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
I = ctypes.c_int64
SOURCE = ROOT / "src" / "capi.mojo"
SHIPPED = "comptime PARALLEL_WORD_THRESHOLD = 1 << 24"
FORCED = "comptime PARALLEL_WORD_THRESHOLD = 0"


def build(threshold_line: str, name: str) -> ctypes.CDLL:
    """Build one variant; return it with a c_int64 restype holder.

    A CDLL whose restype is the shared ``I`` object loses the pointer that owns
    that type: the process then segfaults the next time any ctypes call using
    ``I`` inspects it.  Bind a per-library type instead.
    """
    text = SOURCE.read_text()
    if SHIPPED not in text:
        raise SystemExit(f"{SOURCE}: shipped threshold line not found verbatim")
    scratch = Path(tempfile.mkdtemp())
    src = scratch / "capi.mojo"
    src.write_text(text.replace(SHIPPED, threshold_line))
    out = scratch / f"{name}.so"
    result = subprocess.run(["bash", "-c", f"mojo build --emit shared-lib {src} -o {out}"],
                            capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(result.stderr[-4000:])
    lib = ctypes.CDLL(out)
    lib.mbx_bits_set_range.argtypes = [I] * 4
    lib.mbx_bits_count_range.argtypes = [I] * 3
    lib.mbx_bits_count_range.restype = I
    lib.mbx_bits_binary.argtypes = [I] * 4
    lib.mbx_bits_invert.argtypes = [I] * 2
    lib.mbx_bits_next.argtypes = [I] * 4
    return lib


def words_for(bits: int) -> int:
    return (bits + 63) // 64


def main() -> int:
    serial = build(SHIPPED, "serial")
    parallel = build(FORCED, "parallel")
    print(f"Machine: {platform.node()} | {platform.platform()}")
    print(f"serial build: threshold {SHIPPED!r}")
    print(f"parallel build: threshold {FORCED!r}")

    # Sizes that straddle the real threshold, and the worker granularity edge cases.
    sizes = sorted({
        (1 << 24) * 64 + 7,          # just over the shipped threshold, non-multiple of 64
        (1 << 24) * 64 * 2,          # exactly 2x, word aligned
        (1 << 24) * 64 * 3 + 63,     # odd tail word, ends mid-word
        997 + 64 * ((1 << 24) * 4),  # unaligned start in later set_range tests
    })
    failures = 0
    print()
    print("| size (words) | op | equal |")
    print("| ---: | --- | --- |")

    def check(label: str, size: int, op: str, a: np.ndarray, b: np.ndarray, result=None):
        nonlocal failures
        ok = np.array_equal(a, b) and (result is None or result[0] == result[1])
        if not ok:
            failures += 1
        print(f"| {size} | {label} | {'yes' if ok else 'NO'} |")
        return result

    for bits in sizes:
        words = words_for(bits)

        # set_range with a partial first word, an odd length, and a partial last word.
        start = 37
        count = (bits - 37) // 3 * 2 + 11
        a = np.zeros(words, dtype=np.uint64)
        b = np.zeros(words, dtype=np.uint64)
        serial.mbx_bits_set_range(a.ctypes.data, start, count, 1)
        parallel.mbx_bits_set_range(b.ctypes.data, start, count, 1)
        check("set_range set", words, "set", a, b)
        serial.mbx_bits_set_range(a.ctypes.data, start, count, 0)
        parallel.mbx_bits_set_range(b.ctypes.data, start, count, 0)
        check("set_range clear", words, "clear", a, b)

        # count_range is serial at every size, so both builds must agree trivially;
        # it stays in this list to prove the parallel pass did not disturb it.
        a[:] = 0
        a[5:words - 3] = 0x0123_4567_89AB_CDEF
        r_serial = serial.mbx_bits_count_range(a.ctypes.data, 0, bits)
        r_parallel = parallel.mbx_bits_count_range(a.ctypes.data, 0, bits)
        ok = r_serial == r_parallel
        failures += 0 if ok else 1
        print(f"| {words} | count_range | {'yes' if ok else 'NO'} |")
        window_start, window_count = 71, bits - 200
        r1 = serial.mbx_bits_count_range(a.ctypes.data, window_start, window_count)
        r2 = parallel.mbx_bits_count_range(a.ctypes.data, window_start, window_count)
        ok = r1 == r2
        failures += 0 if ok else 1
        print(f"| {words} | count_range (unaligned window) | {'yes' if ok else 'NO'} |")

        # iand / ior / ixor against a patterned operand.
        mask = np.zeros(words, dtype=np.uint64)
        mask[1::3] = 0xF0F0_F0F0_F0F0_F0F0
        mask[::7] = 0xFFFF_FFFF_FFFF_FFFF
        for op, label in ((0, "iand"), (1, "ior"), (2, "ixor")):
            a = np.full(words, 0x00FF_00FF_00FF_00FF, dtype=np.uint64)
            b = a.copy()
            serial.mbx_bits_binary(a.ctypes.data, mask.ctypes.data, bits, op)
            parallel.mbx_bits_binary(b.ctypes.data, mask.ctypes.data, bits, op)
            check(label, words, label, a, b)

        # invert, with a partial trailing word that must not gain set high bits.
        a = np.full(words, 0x0123_4567_89AB_CDEF, dtype=np.uint64)
        b = a.copy()
        serial.mbx_bits_invert(a.ctypes.data, bits)
        parallel.mbx_bits_invert(b.ctypes.data, bits)
        check("invert", words, "invert", a, b)
        tail_bits = bits % 64
        if tail_bits:
            ok = int(a[-1]) < (1 << tail_bits) and int(b[-1]) < (1 << tail_bits)
            failures += 0 if ok else 1
            print(f"| {words} | invert tail bits {tail_bits} masked | {'yes' if ok else 'NO'} |")

    print()
    print("PASS" if failures == 0 else f"FAIL: {failures} mismatches")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
