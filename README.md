# mojo-bx-python

`mojo-bx-python` is a standalone Mojo port of the compute-heavy interval and
alignment utilities in [bx-python](https://github.com/bxlab/bx-python).  It
keeps the covered `bx.*` import paths and call signatures so existing callers
can move a covered workload by putting this repository's `python/` directory
ahead of bx-python on `PYTHONPATH`.

## Covered subset

| upstream module | covered API |
| --- | --- |
| `bx.intervals.intersection` | `Interval`, `IntervalNode`, `IntervalTree`, `Intersecter`, half-open `find`, before/after, interval shortcuts, strand-aware upstream/downstream, `traverse` |
| `bx.bitset` | `BitSet`, `BinnedBitSet`, range set/clear/count, searches, boolean operations, indexing and bounds checks |
| `bx.align.score` | `ScoringScheme`, scoring-matrix builders/readers, `score_texts`, `score_alignment`, `accumulate_scores`, `hox70` |

`overlap_counts(starts, ends, query_starts, query_ends)` is also provided in
`bx.intervals.intersection` for fast batched, half-open overlap counts.

Not included are BED/MAF/AXT/LAV readers and writers, interval file indexes,
clustering, PWM utilities, EPO, site masks, and alignment manipulation tools.
`BinnedBitSet` uses the same packed 64-bit-word storage as `BitSet`, rather
than bx-python's sparse bins: its methods and results match, but it does not
offer the upstream long-run compression advantage.

## Install

```bash
pixi install
pixi run build
```

The test environment also installs bx-python from PyPI, so parity tests use
the actual upstream implementation. The Python wrapper will rebuild
the shared library automatically if `src/capi.mojo` is newer than it.

## Usage

Run this from the repository root through Pixi:

```bash
pixi run python - <<'PY'
from bx.align.score import hox70, score_texts
from bx.intervals.intersection import Interval, IntervalTree

tree = IntervalTree()
start = len("abcdefghij")
tree.add_interval(Interval(start, start + len("abcdefghij"), value="exon"))
print([x.value for x in tree.find(start, start + len("a"))])
assert score_texts(hox70, "ACGT", "ACGT")
PY
```

## Performance

Measured with `pixi run bench` on `leaf-gpu-dedicated-server`, Linux
6.8.0-136-generic x86_64 (glibc 2.39), Python 3.13.14. Times include the
public Python API but exclude process startup; each case runs both packages in
an isolated interpreter with the same input.

| case | mojo-bx-python | bx-python | result |
| --- | ---: | ---: | --- |
| interval find (20k intervals, 2k queries) | 22.1 ms | 8.4 ms | 2.64x slower |
| alignment scoring (1M columns x 10) | 52.5 ms | 3579.3 ms | 68.13x faster |
| bitset set/count (10M bits x 20) | 7.5 ms | 17.4 ms | 2.33x faster |

The alignment kernel uses one native pass over the encoded columns, eliminating
Python's per-character loop. Bitsets use packed `uint64` words; range, count,
Boolean, and inversion loops use CPU SIMD with masked scalar boundaries and
tails. Mutating bitset operations parallelize only above 8 MiB of packed
storage. Interval queries combine binary bounds and filtering in one FFI call,
reuse native output storage, and gather Python objects through a NumPy object
array. They remain slower than bx-python's mature Cython treap.

No GPU path is included. None of the covered kernels exceeds roughly two
floating-point operations per byte moved: bitsets and interval scans are
memory/branch bound, while alignment scoring is a branch-heavy table lookup.
GPU transfer and launch overhead would therefore lose to the CPU path.

Reproduce the table only through the flock-protected task:

```bash
pixi run bench
```

## How it works

Python owns contiguous NumPy buffers. ctypes passes their raw addresses as
64-bit integers to a single Mojo compilation unit, `src/capi.mojo`; every
export reconstructs a mutable `UnsafePointer` internally. No Mojo function
allocates or retains Python memory. Intervals use start-sorted `int64` start
and end buffers; bit sets use packed `uint64` words; scoring tables are contiguous
row-major `float64` arrays and alignment text is Latin-1 `uint8`.

```text
python/bx/*  ->  ctypes Int addresses  ->  src/capi.mojo  ->  dist/libmojo-bx-python.so
```

## Verification

```bash
pixi run build && pixi run test && pixi run bench
```

The suite includes behavioral parity checks against the installed upstream
package, published scoring vectors, and FFI-boundary regression cases.

## License

MIT
