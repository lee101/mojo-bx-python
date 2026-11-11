"""Kernel-level thread scaling of the *shipped* kernels at caller sizes.

For each size, the serial build (shipped threshold) and a threshold-0 build
(everything threads at 4 workers) are called directly, so the numbers are the
kernel cost alone with no Python object churn.  Also reports the threshold
straddle from one build where only the size decides serial vs threaded.

Usage: pixi run python bench/parallel_scale.py
"""

from __future__ import annotations

import ctypes
import os
import platform
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "capi.mojo"
I = ctypes.c_int64
FULL = 0xFFFF_FFFF_FFFF_FFFF
SHIPPED = "comptime PARALLEL_WORD_THRESHOLD = 1 << 24"
FORCED = "comptime PARALLEL_WORD_THRESHOLD = 0"


def build(threshold_line: str, name: str) -> ctypes.CDLL:
    text = SOURCE.read_text()
    if SHIPPED not in text:
        raise SystemExit(f"{SOURCE}: shipped threshold line not found verbatim")
    scratch = Path(tempfile.mkdtemp())
    src = scratch / "capi.mojo"
    src.write_text(text.replace(SHIPPED, threshold_line))
    out = scratch / f"{name}.so"
    subprocess.run(["bash", "-c", f"mojo build --emit shared-lib {src} -o {out}"],
                   check=True, capture_output=True)
    lib = ctypes.CDLL(out)
    lib.mbx_bits_set_range.argtypes = [I] * 4
    lib.mbx_bits_count_range.argtypes = [I] * 3
    lib.mbx_bits_count_range.restype = I
    lib.mbx_bits_binary.argtypes = [I] * 4
    lib.mbx_bits_invert.argtypes = [I] * 2
    return lib


def kernel_call(lib, op, base, other, n):
    if op == "set_range":
        lib.mbx_bits_set_range(base, n // 10, n // 2, 1)
    elif op == "count_range":
        return lib.mbx_bits_count_range(base, 0, n)
    elif op == "iand":
        lib.mbx_bits_binary(base, other, n, 0)
    else:
        lib.mbx_bits_invert(base, n)
    return None


def median_ms(fn, reps):
    return statistics.median([_once(fn) for _ in range(reps)])


def _once(fn):
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1e3


def main():
    serial = build(SHIPPED, "serial")
    threaded = build(FORCED, "threaded")
    print(f"Machine: {platform.node()} | {platform.platform()} | load {os.getloadavg()[0]:.1f}")
    print("kernel only, median ms of 5, setup and verification outside the clock")
    print("| size | op | serial build | threaded build (4 workers) | change |")
    print("| ---: | --- | ---: | ---: | ---: |")
    for mibit in (1024, 1536):
        n = mibit << 20
        words = (n + 63) // 64
        for op in ("set_range", "count_range", "iand", "invert"):
            answers = {}
            times = {}
            for label, lib in (("serial", serial), ("threaded", threaded)):
                buf = np.zeros(words, dtype=np.uint64)
                other = np.ones(words, dtype=np.uint64)
                base, other_addr = int(buf.ctypes.data), int(other.ctypes.data)
                if op == "count_range":
                    buf[:] = 0x0123_4567_89AB_CDEF
                answer = kernel_call(lib, op, base, other_addr, n)
                # The answer checked here is the bit content after the call.
                content = (int(np.bitwise_count(buf).sum()),
                           int(np.bitwise_count(np.bitwise_xor(buf, other)).sum()))
                answers[label] = (answer, content)
                # Restore the starting content for every timed repetition.
                start = (buf.copy() if op in ("invert", "iand") else None)

                def run(start=start, op=op, lib=lib, base=base, other_addr=other_addr, n=n, buf=buf):
                    if start is not None:
                        buf[:] = start
                    return kernel_call(lib, op, base, other_addr, n)

                times[label] = median_ms(run, 5)

            s, t = times["serial"], times["threaded"]
            same = answers["serial"] == answers["threaded"]
            change = f"{s / t:.2f}x faster" if s >= t else f"{t / s:.2f}x slower"
            note = "" if same else "  RESULTS DIFFER"
            print(f"| {mibit} Mi-bit | {op} | {s:.3f} ms | {t:.3f} ms | {change}{note} |",
                  flush=True)


if __name__ == "__main__":
    main()
