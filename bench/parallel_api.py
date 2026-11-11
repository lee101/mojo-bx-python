"""Force the parallel branch at a caller size, through the public Python API.

Used to compare serial and threaded builds at one size (e.g. 1024 Mi-bit, just
reporting the ``e`` of the ``e``-level threshold).

Usage: pixi run python bench/parallel_api.py <threshold: shipped|forced> [sizes]
"""

from __future__ import annotations

import ctypes
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "capi.mojo"
I = ctypes.c_int64
SHIPPED = "comptime PARALLEL_WORD_THRESHOLD = 1 << 24"
FORCED = "comptime PARALLEL_WORD_THRESHOLD = 0"


def build(threshold_line: str, name: str) -> ctypes.CDLL:
    text = SOURCE.read_text()
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


def main():
    sys.path.insert(0, str(ROOT / "python"))
    from bx import bitset

    libs = {"serial": build(SHIPPED, "serial"), "threaded": build(FORCED, "threaded")}
    sizes = [int(x) for x in (sys.argv[1:] or ["1024"])]
    ops = ("set_range", "count_range", "iand", "invert")
    results = {}
    for label, lib in libs.items():
        bitset._NATIVE = lib
        results[label] = {}
        for mibit in sizes:
            n = mibit << 20
            row, answers = [], []
            for op in ops:
                samples = []
                expected = None
                for _ in range(5):
                    # Setup is outside the clock: it is common to both builds.
                    b = bitset.BitSet(n)
                    o = None
                    if op in ("set_range", "count_range"):
                        b.set_range(n // 10, n // 2)
                    elif op == "iand":
                        o = bitset.BitSet(n)
                        o.set_range(0, n // 3)
                        b.set_range(0, n // 2)
                    else:
                        b.set_range(0, n // 2)
                    t0 = time.perf_counter()
                    if op == "set_range":
                        b.set_range(n // 10, n // 2)
                    elif op == "count_range":
                        b.count_range(0, n)
                    elif op == "iand":
                        b.iand(o)
                    else:
                        b.invert()
                    samples.append((time.perf_counter() - t0) * 1e3)
                    got = b.count_range(0, n)
                    if expected is None:
                        expected = got
                    assert got == expected, f"{label} {op} not reproducible"
                row.append(statistics.median(samples))
                answers.append(expected)
            results[label][mibit] = (row, answers)

    # The threaded build must produce exactly the serial build's answers.  This
    # runs *first*, on a buffer both builds have already written at threshold 0,
    # so it also proves a threaded write matches the serial one on the same bytes.
    for mibit in sizes:
        serial_answers = results["serial"][mibit][1]
        threaded_answers = results["threaded"][mibit][1]
        assert serial_answers == threaded_answers, (
            f"{mibit} Mi-bit: threaded and serial builds disagree: "
            f"{serial_answers} vs {threaded_answers}")
    print(f"identical results in both builds for every size: {sizes}")
    print("| size | op | serial build | threaded build | change |")
    print("| ---: | --- | ---: | ---: | ---: |")
    for mibit in sizes:
        for index, op in enumerate(ops):
            s = results["serial"][mibit][0][index]
            t = results["threaded"][mibit][0][index]
            change = f"{s / t:.2f}x faster" if s >= t else f"{t / s:.2f}x slower"
            print(f"| {mibit} Mi-bit | {op} | {s:.3f} ms | {t:.3f} ms | {change} |", flush=True)


if __name__ == "__main__":
    main()
