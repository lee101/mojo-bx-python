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
6.8.0-136-generic x86_64 (glibc 2.39) then 6.8.0-147-generic, Python 3.13.14.
Times include the public Python API but exclude process startup; each case runs
both packages in an isolated interpreter with the same input.

| case | mojo-bx-python | bx-python | result |
| --- | ---: | ---: | --- |
| interval find (20k intervals, 2k queries) | 27.8 ms | 8.4 ms | 2.64x slower |
| alignment scoring (1M columns x 10) | 52.5 ms | 3579.3 ms | 68.13x faster |
| bitset set/count (10M bits x 20) | 7.5 ms | 17.4 ms | 2.33x faster |
| ^ re-measured before the unroll pass | 27.8 / 46.8-52.5 / 7.5-8.5 ms | - | - |
| ^ re-measured before the SIMD pass | 29.5 / 76.3 / 7.1 ms | 10.3 / 3733.9 / 21.3 ms | - |
| ^ re-measured after the SIMD pass | 23.0 / 36.5 / 7.5 ms | 9.1 / 3527.8 / 18.3 ms | see the A/B table below |

The unroll pass kept no kernel change. The SIMD pass re-measured `pixi run
bench` before and after (the last two `^` rows): the alignment case moved 76.3
-> 36.5 ms because of the `_native_table` change described below, and the other
two stayed inside this host's spread. Because that spread is large, every
candidate was also A/B measured against a saved baseline library, alternating
between the two in one process:

| case | before | after | result |
| --- | ---: | ---: | --- |
| `overlap_counts` (20k intervals, 2k queries), Python API | 15.85-16.16 ms | 0.65-0.80 ms | 20-24x faster |
| `overlap_counts`, kernel only (same C ABI call) | 15.79 ms | 0.52-0.55 ms | 29-30x faster |
| `score_texts` 1M x1, Python API | 4.07 / 4.14 / 5.66 ms | 3.30 / 3.41 / 2.38 ms | 1.19-1.25x faster |
| `score_texts` 1M x10, Python API | 41.59 / 41.59 / 41.78 ms | 33.38 / 33.31 / 23.73 ms | 1.25-1.76x faster |
| `score_texts` 1M, no gaps, Python API | 2.71 / 2.71 / 2.71 ms | 2.09 / 2.10 / 2.06 ms | 1.29-1.31x faster |
| `accumulate_scores` 1M, Python API | 2.98 / 2.96 / 3.01 ms | 2.48 / 2.31 / 2.31 ms | 1.20-1.31x faster |
| `score_texts` / `accumulate_scores`, raw 1M call (kernel) | 2.10 / 2.03 ms | 2.10 / 2.03 ms | unchanged |
| raw kernel, all-gaps `score_texts` (1M) | 0.724 ms | 0.724 ms | unchanged |
| `filter_indices` alone (170 windows x 2000) | 0.32-0.34 ms | 0.14-0.15 ms | 2.3x faster |
| `IntervalTree.find` (20k intervals, 2k queries), Python API | 12.64 / 12.54 / 12.71 ms | 13.61 / 13.80 / 12.76 ms | 0.93-1.00x, no change detected |
| bitset `set_range` + `count_range` (10M bits), Python API | 0.209 / 0.213 / 0.249 ms | 0.249 / 0.212 / 0.209 ms | no change detected |

Kernel rows are the fastest of 9 alternating repetitions between a saved
baseline `.so` and the candidate, pinned with `taskset -c 61` because this is a
shared 72-core host. Python-API rows compare the git-`HEAD` wrappers against the
baseline `.so` with the working-tree wrappers against the candidate `.so`, each
in its own process, best of 7 over three runs. The scoring rows are the one
Python-level change kept: `_native_table` used to scan the whole table through
`astype(object)` on every call, and now only does so for 64-bit integer dtypes,
which cannot be skipped without losing the 2**53 exactness check. The kernel
itself is unchanged, which is why the raw-call rows are identical and only the
wrapped call moves.

Kept: `mbx_overlap_counts`. Its inner loop had a data-dependent `break` on the
start-sorted `starts` and a scalar compare, so it was neither vectorisable nor
cache-friendly: each of the 2000 queries rescanned from index 0. Replacing the
break with `lower_bound(starts, query_end)` and adding an
`upper_bound(max_ends, query_start)` skip makes every remaining iteration
independent, so it is now a 4-lane `int64` compare-and-`reduce_add` with a
scalar tail. Split three ways on the real inputs (2000 queries, 20000
intervals): old scan 16.4-18.1 ms, bounds + scalar count 0.59-0.72 ms, bounds +
SIMD count 0.44-0.56 ms. So the bound pruning is the bulk of it (170-candidate
windows become ~35-element slices) and the SIMD reduction is 1.1-1.35x on top,
which is what shows up as the 29-30x kernel row. Exactness is unchanged:
the skipped prefix provably cannot contain an overlap, and the counts compared
equal to the baseline on the real inputs and on the boundary tests.
`filter_indices` (the compaction behind `IntervalTree.find` / `find_bounded`)
was vectorised the same way. Its kernel halves, but the `IntervalTree.find`
caller case did not move because it is dominated by the per-query Python and
ctypes cost, so its end-to-end result is reported as unchanged rather than
claimed as a win.

Reverted: three SIMD shapes for the alignment scan, all exact and all slower
than the scalar state machine at the caller size (1M columns):

| candidate | effect | kept |
| --- | --- | --- |
| SIMD classify (`eq`/`ne` masks) + scalar lane loop | 6.4-6.5 ms vs 3.0 ms | no |
| SIMD classify + conditional per-lane table lookup | 19.0 ms vs 3.0 ms | no |
| SIMD classify + unconditional per-lane lookup, masked `reduce_add` | 19.1 ms vs 3.0 ms | no |

The reason is structural, and it is the same reason the unroll pass above
failed: a column's `gap_open`/`gap_extend` depends on the class of the
immediately preceding non-both-gap column, so the carry cannot be reduced in a
lane. Widening only triples the work per column (compress masks back to scalars
and walk them), and the table lookup is a gather, which AVX2 does slowly. The
scalar loop the compiler emits is already three dependent compares and one load
per column. Bitset kernels were left alone: they already use 256-bit
`uint64` loops with a scalar tail, and at 0.153 ms per 10M bits (1.25 MB of
packed words) they are at memory speed. `mbx_accumulate_scores` writes an
output value per column and carries the same state, so it has the same
constraint as `score_texts`.

Boundary coverage for every native block loop added in this pass (lengths 0, 1,
`W-1`, `W`, `W+1`, and 1003, plus an unaligned base pointer) lives in
`tests/test_simd_boundaries.py`. The FFI buffers are zero-copy NumPy views of
Python `bytes`/arrays, so their base addresses are only 16-byte aligned in
practice (measured 48 mod 64 for a 1 MB text); the kernels therefore use
element loads (`unsafe_load` / `load[width=W]`) and never assume more.

The unroll pass candidates (built, tested and A/B measured against a saved
baseline library first):

| candidate | isolated effect | caller effect | kept |
| --- | --- | --- | --- |
| `mbx_score_texts`, 4 unrolled lanes, independent match sums | ~1.4 ms -> ~0.79 ms per 1M columns | slower once gap state restored | no |
| `mbx_score_texts`, 4 chunked passes with cross-chunk gap carry | within noise | 2 parity vectors failed, then slower and still inexact | no |
| `mbx_bits_count_range`, 4 partial popcount counters | ~104 -> ~78 us per 10M bits | inside noise | no |
| `mbx_bits_count_range`, 8 / 16 / 24 / 32 chains | 73 / 63 / 150 / - us | inside noise, or slower | no |

The isolated numbers are quartile minima over 200 repeated calls; the caller
case is `pixi run bench`'s own bitset case, which includes the allocation and
`set_range` around the count. That case is bounded by touching 1.25 MB of
packed words: at ~46 GB/s here, the count alone can never account for its
10 ms, so shaving the kernel does not move it.

Two facts make further chain-splitting unattractive here. First, a gap's
`gap_open`/`gap_extend` depends on the immediately preceding column, whether
that column is a two-sided gap, and which side the gap is on, so no column's
penalty can be deferred to a combine step without changing the result; every
exact split pays for the bookkeeping and every cheap split changes results.
Second, the scoring sum is over integer table entries, so a multi-accumulator
split would only matter if a single `Float64` accumulator lost precision, which
it does not at these magnitudes. If a variant that splits the sum is adopted
later, results can differ in the last bits; the parity tolerance must then
stay as it is, and the README must say so. `@parameter for` is not available on
this pinned compiler (`MOJO_NOTES.md` section 1), so fixed trip counts were
written out.

The alignment kernel uses one native pass over the encoded columns, eliminating
Python's per-character loop. Bitsets use packed `uint64` words; range, count,
Boolean, and inversion loops use CPU SIMD with masked scalar boundaries and
tails. Mutating bitset operations parallelize only above 8 MiB of packed
storage. Interval queries combine binary bounds and filtering in one FFI call:
`overlap_counts` binary-searches both ends of the candidate window and counts
the survivors with a SIMD reduction, and `find_bounded` compacts a SIMD block at
a time. Queries reuse native output storage and gather Python objects through
a NumPy object array. `find` remains slower than bx-python's mature Cython
treap because its wall clock is per-query Python and ctypes cost, not the
kernel. Scoring tables are only range-checked for 64-bit integer dtypes, since
narrower integers cannot hold a value above 2**53, which keeps that scan off
the per-call path. `accumulate_scores` and `score_texts` keep their sequential
gap carry in the kernel for the reason given above.

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
