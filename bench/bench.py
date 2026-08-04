"""Reproducible benchmarks against the installed bx-python package."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CASES = {
    "interval find (20k intervals, 2k queries)": """import time, json
from bx.intervals.intersection import IntervalTree
t=IntervalTree()
for i in range(20000): t.insert(i*3, i*3+11, i)
t0=time.perf_counter()
for q in range(2000): t.find(q*23, q*23+500)
print(json.dumps((time.perf_counter()-t0)*1000))""",
    "alignment scoring (1M columns x 10)": """import time, json
from bx.align.score import hox70, score_texts
a=('ACGT-'*200000); b=('AGGT-'*200000)
t0=time.perf_counter()
for _ in range(10): score_texts(hox70,a,b)
print(json.dumps((time.perf_counter()-t0)*1000))""",
    "bitset set/count (10M bits x 20)": """import time, json
from bx.bitset import BitSet
t0=time.perf_counter()
for _ in range(20):
 b=BitSet(10_000_000); b.set_range(1_000_000,8_000_000); b.count_range(0,10_000_000)
print(json.dumps((time.perf_counter()-t0)*1000))""",
}


def run(code, local):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "python") if local else ""
    return json.loads(subprocess.run([sys.executable, "-c", code], cwd="/tmp", env=env,
                                     text=True, check=True, capture_output=True).stdout)


def main():
    print(f"Machine: {platform.node()} | {platform.platform()} | Python {platform.python_version()}")
    print("| case | mojo-bx-python | bx-python | result |")
    print("| --- | ---: | ---: | --- |")
    for name, code in CASES.items():
        mojo, upstream = run(code, True), run(code, False)
        ratio = upstream / mojo
        result = f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.2f}x slower"
        print(f"| {name} | {mojo:.1f} ms | {upstream:.1f} ms | {result} |")


if __name__ == "__main__":
    main()
