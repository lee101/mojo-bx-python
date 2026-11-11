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
tails. Interval queries combine binary bounds and filtering in one FFI call:
`overlap_counts` binary-searches both ends of the candidate window and counts
the survivors with a SIMD reduction, and `find_bounded` compacts a SIMD block at
a time. Queries reuse native output storage and gather Python objects through
a NumPy object array. `find` remains slower than bx-python's mature Cython
treap because its wall clock is per-query Python and ctypes cost, not the
kernel. Scoring tables are only range-checked for 64-bit integer dtypes, since
narrower integers cannot hold a value above 2**53, which keeps that scan off
the per-call path. `accumulate_scores` and `score_texts` keep their sequential
gap carry in the kernel for the reason given above.

### Parallelism

Packed-word bitset loops run on worker threads above a size threshold, using
`parallelize` from `max.algorithm` (`MOJO_NOTES.md` section 4). The call always
passes an explicit worker count: the 2-argument form segfaults from a ctypes
host, which `bench/probe_parallel_host.py` reproduces.

Three mutations thread: `set_range`, the `iand`/`ior`/`ixor` family, and
`invert`. Each splits its word range into `PARALLEL_TASKS = 4` contiguous
chunks rounded up to a whole SIMD block, so no lane ever straddles a chunk
boundary. `count_range` and every interval, alignment and scoring kernel stay
serial (see the reverted list below).

The threshold is `PARALLEL_WORD_THRESHOLD = 1 << 24` words: 128 Mi-bit, or
16 MiB of packed storage. It was chosen from measurements, not guessed:

* A threaded `parallelize` launch has a fixed cost on this host of roughly
  2.8-7.2 ms (median of 200 empty-worker launches: 4.5-5.2 ms for 2-8 workers,
  spread by shared-host load, and independent of worker count). A serial loop
  over the same number of units costs about 1-2 us. Well below the threshold
  the serial loop finishes all its work in less than the launch costs, so
  threading can only lose there.
* At caller sizes the crossover measured between 128 and 1024 Mi-bit: at
  128 Mi-bit threading was 1.00-1.05x, at 1024 Mi-bit `set_range` ranged from
  parity to 1.24x slower depending on host load, and `iand`/`invert` were
  1.06-1.40x faster.
* Only far above the caller range is the win large and stable. At 8192 Mi-bit,
  repeated verified trials give `iand` 1.54-1.67x and `invert` 1.26-1.39x.

So the threshold sits at a conservative 16 MiB rather than at the measured
crossover: threading is a real win well above it, host noise swamps it near the
crossover, and below it the launch would dominate. Sizes between the threshold
and the crossover (16-128 MiB of packed words) pay a modest loss; that is the
measured margin above the crossover point rather than a claim that threading
wins everywhere above the cutoff.

Thread scaling was measured at 1, 2, 4, 6 and 8 workers
(`bench/thread_scale.py`, `bench/thread_scale_large.py`) on 1 GiB and 4 GiB
calls:

| size, kernel | 1 | 2 | 4 | 6 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 GiB `count_range` read | 0.98x | 1.20x | 1.38x | 1.44x | 1.46x |
| 1 GiB `iand` | 1.03x | 1.45x | 1.67x | 1.53x | 1.80x |
| 1 GiB `invert` | 1.01x | 1.22x | 1.39x | 1.36x | 1.38x |
| 4 GiB `set_range` | 1.12x | 1.26x | 1.37x | 1.42x | 1.47x |
| 4 GiB `count_range` read | 1.03x | 1.27x | 1.44x | 1.53x | 1.55x |
| 4 GiB `iand` | 0.99x | 1.23x | 1.57x | 1.64x | 1.72x |

Scaling flattens after 4 workers (4 -> 6 adds under 10%, 6 -> 8 is inside the
noise and occasionally reverses, as `iand` at 1 GiB does). The cap is therefore
4, which also respects the shared host.

Before and after, with the production API on 1024 and 1536 Mi-bit bitsets (the
largest `BitSet` the library accepts) and everything else equal. "Before" is the
git-`HEAD` library and "after" is this one; setup is outside the clock, results
are asserted identical between builds, and each cell is the median of 5 within
one batch. Six batches were taken; the ranges below are across batches, because
this shared host's load moved between 14 and 35 while they ran
(`bench/parallel_api.py`, `bench/parallel_scale.py`):

| size, op | before | after | result |
| --- | ---: | ---: | --- |
| 1024 Mi-bit, `iand` | 71.4-136.6 ms | 52.9-67.5 ms | 1.06-2.58x faster |
| 1024 Mi-bit, `invert` | 73.8-122.4 ms | 52.8-62.7 ms | 1.20-1.95x faster |
| 1536 Mi-bit, `iand` | 98.7-187.9 ms | 85.2-168.9 ms | 1.11-1.23x faster |
| 1536 Mi-bit, `invert` | 95.2-166.9 ms | 82.6-149.2 ms | 1.09-1.24x faster |
| 1024 Mi-bit, `count_range` | 16.5-41.6 ms | 15.8-35.8 ms | 1.00x (serial in both) |
| 1536 Mi-bit, `count_range` | 24.0-61.0 ms | 24.4-44.6 ms | 1.00x (serial in both) |
| 1024 Mi-bit, `set_range` | 9.7-37.3 ms | 16.7-30.0 ms | 1.72-1.96x slower (forced) |
| 1536 Mi-bit, `set_range` | 14.4-60.8 ms | 16.2-31.6 ms | 1.17x slower to 1.80x faster |

The single widest cell (`iand`, 1.06-2.58x) is reported as measured, but the
tight repeatable range is the narrower one in the kernel-only runs; treat the
outer edges as host noise, not as a scalable figure.

`set_range` fill behaviour, stated plainly because the numbers disagree across
batches. Measured on the final build with fresh 4 MiB `BitSet` allocations per
repetition (median of 15, serial build vs forced-threaded build):

| size | serial | threaded | outcome |
| ---: | ---: | ---: | --- |
| 1024 Mi-bit | 10.6 ms | 12.1 ms | 1.14x slower |
| 1536 Mi-bit | 15.4 ms | 15.9 ms | 1.03x slower |
| 8192 Mi-bit | 49.7 ms | 52.1 ms | 1.05x slower |
| 20480 Mi-bit | 89.2 ms | 91.5 ms | 1.03x slower |

Earlier batches on the same mutation ranged from roughly parity to 1.4x
faster (at 1536 Mi-bit) and to 1.5-1.9x slower, so the sign of this effect is
not stable on this host: the host's load moved between 14 and 35 during these
measurements. The defensible reading is that threading the fill is never a
clear, repeatable win at any size that was measured, and the win for the same
mutation reported at 20+ GiB in one earlier run did not reproduce against the
final build. Because the fill shares the threshold with `iand`/`invert`, a
caller above 1536 Mi-bit may pay up to about 1.1x on the fill for the
1.06-1.40x measured on `iand` and 1.09-1.31x on `invert` at caller sizes (the
large-size thread-scaling runs give 1.7x and 1.4x). The fill is not reverted
to serial because, with a shared cutoff, the three operations still come out
ahead together, but it is the weakest of the three and should be re-measured if
this host becomes quieter. The shipped build keeps the serial fill at
1024 Mi-bit (16 MiB is under the 1 GiB threshold line), so the exposure starts
above 1536 Mi-bit.

Also kept, below the threshold: `set_range` fills the word range with 256-bit
stores (4 x 32 B per 64 B line). It wins 5-40% on fresh storage at 0.4-2.8 ms
and measures parity at 30 ms.

Not hidden either: the alignment and interval kernels were re-probed for
parallelism and left serial. Their per-query and per-column work carries state
(gap open/extend across the preceding column) that a chunk boundary would
change the result for; splitting them needs bookkeeping that costs more than the
credit earned, which is the same conclusion the SIMD pass reached above. The
bitset count read threads well on its own (3.5-3.7x on 4 workers at 4 GiB), but
`count_range` is left serial: the variants that keep each worker's data in cache
while counting it lost at the sizes a caller reaches, their scratch buffers
polluted other buffers' locality, and the reduction into a shared partial
buffer has no measurable gain.

Reverted, each measured before being dropped:

| candidate | measured | outcome |
| --- | --- | --- |
| interleave blocks across the calling thread (`parallelize(f, parts, 1)`) | 0.95-1.25x, worse than one launch | reverted |
| threaded `count_range` with a two-stage grid, 16/64 MiB chunks | 1.00-1.05x on the read, 1.08x at 1024 Mi-bit, 1.01x at 1536 Mi-bit | reverted |
| threaded counting via cache-sized prefetch + per-worker scratch | 0.33x, prefetch evicted the loop's own data | reverted |
| threaded fill with the 256-bit serial body, at 1024-1536 Mi-bit | 1.5x slower than serial | reverted |
| `binary_words_wide` 256-bit body for `iand`/`ior`/`ixor` | 1.05x slower at 128 Mi-bit | reverted |
| `set_range`/`count_range` threading at 1024-1536 Mi-bit | 2.79x/5.36x slower in the API batch once setup was excluded | not reverted, see below |

`count_range` is reverted outright: threading it never beat serial by more
than 8%, and its reduction also needs a scratch buffer. The threaded fill is not
reverted: on repeated trials it paid at 1024-1536 Mi-bit (see the before/after
table; a different host-load window shows parity), and it stays under the one
shared threshold for the `iand`/`invert` wins, with its loss reported there.
Every candidate above was measured alternating with the baseline on the same
buffer in one process.

Equivalence is tested, not assumed: `tests/test_parallel_paths.py` builds
`src/capi.mojo` twice (shipped threshold, and threshold forced to 0 so every
mutation threads even at test-sized inputs) and compares 16, 192 and
1536 Mi-bit results bit-for-bit, including unaligned first words, partial last
words, tail-bit masking after `invert`, and the same operations through the
public `BitSet` API against a swapped native library. The module-level
equivalence harness (`bench/parallel_equivalence.py`) additionally checks 32-64
Mi-word sizes and all three boolean operations. Because the branch is decided
by size alone, a threaded run at 1536 Mi-bit exercises exactly the code a
threaded run at 8192 Mi-bit would.

## GPU

CPU only: memory-bound. No GPU path is shipped, and none is planned for this
subset.

The GPU gate for this phase was run rather than assumed. Two things are on
record. First, `bench/analysis.md` does not exist in this repository (it is not
in any commit either), so the kernel list was rebuilt from the code itself:
eleven exports in `src/capi.mojo`, measuring flops against bytes read *and*
written per element. Second, the shared device was free enough to use:

```
$ nvidia-smi --query-gpu=memory.free --format=csv,noheader
5190 MiB            # RTX 5090, above the 4000 MiB floor
$ pixi run mojo run bench/probe_gpu.mojo
device: NVIDIA GeForce RTX 5090
```

So the phase was not skipped for a missing device; it was rejected on the
arithmetic-intensity gate. Per-element arithmetic intensity of every covered
kernel:

| kernel | work per element | bytes per element | intensity | gate |
| --- | --- | ---: | ---: | --- |
| `bits_set_range`, `bits_binary`, `bits_invert` | 1 bitwise op per 8-bit word | 16 (read+write) | ~0.06 op/B | memory-bound |
| `bits_count_range` | 1 popcount per word | 8 | ~0.25 op/B | memory-bound |
| `find_indices*`, `overlap_counts` | 1-2 integer compares | 8-16 | ~0.1-0.25 op/B | memory/branch-bound |
| `score_texts`, `accumulate_scores` | 1 table load + ~4 compares + 1 FP add | 2 streamed (table in L1) | ~3 op/B but a serial carry | not splittable |

Only `score_texts`/`accumulate_scores` clear two operations per byte, and only
if integer compares are counted as operations. They are also the one kernel
that cannot use a GPU meaningfully: `gap_open`/`gap_extend` depend on the class
of the immediately preceding column, so the kernel is a strictly sequential
state machine. A GPU would run it as a single thread over every column — the
"one kernel launch" rule cannot help when there is only one thread's worth of
parallel work. This is the same wall the SIMD pass and the unroll pass hit
(see "Reverted" above), now for the third time.

Measured, not estimated (`bench/probe_intensity.py`, `bench/probe_gpu_floor.py`;
best of 9 calls per case, two runs on the shared host):

| case | CPU | GPU floor (copy up + copy back + sync, no kernel) | result |
| --- | ---: | ---: | --- |
| `score_texts` raw, 1 M cols (2 MB) | 2.1-2.3 ms | 0.54 + 1.15 ms | copy alone ~0.7x the whole call; a kernel only adds |
| `score_texts` raw, 100 M cols (200 MB) | 217-219 ms | 31 + 51-61 ms = 82-92 ms | GPU floor is ~2.4-2.7x under CPU, but the kernel is one serial thread, so the launch cannot help |
| bitset `set_range`, 10 M bits (1.25 MB) | 0.055 ms | > 1.7 ms at 1 MB | GPU loses by >30x before any kernel |
| bitset `count_range`, 10 M bits | 0.105-0.204 ms | > 1.7 ms at 1 MB | GPU loses by >8x before any kernel |
| bitset `iand`, 10 M bits | 0.109-0.205 ms | > 1.7 ms at 1 MB | GPU loses by >8x before any kernel |
| `overlap_counts` raw, 20k intervals x 2k queries | 0.51-0.73 ms | 1 MB round trip is already > 1.7 ms | GPU loses; the CPU path is bounds + SIMD over ~35 candidates per query |

The 1 MB floor row is the smallest transfer measured (0.54 ms up, 1.15-1.31 ms back);
the 200 MB and 1 GiB rows scale roughly linearly (200 MB: 31 ms up, 51-61 ms
back; 1 GiB: 198-203 ms up, 389-419 ms back, i.e. about 5.5 GB/s round trip on
this shared device). The bitset cases
move 1.25 MB of packed words and would pay a full round trip for a kernel that
already runs at 23-45 GB/s on the CPU. The only case where the raw copy is
cheaper than the CPU call is the 100 M-column alignment, and there the work is
one strictly dependent chain of columns: `gap_open`/`gap_extend` for column *i*
depends on the gap state of column *i-1*, so the maximum parallelism is one
thread. One GPU thread over 100 M columns cannot beat a CPU core doing 2.2 ns
per column, and the CPU number already includes the whole call.

Nothing was built and reverted: no candidate passed the flops/byte gate, so no
kernel was written and no A/B table of GPU variants exists to report. The CPU
kernels are unchanged by this phase. The device-allocation limit (2 GiB) was
never approached: the largest buffer the probes allocated was 1 GiB, and the
smallest free-memory reading taken before a run was 5190 MiB.

`tests/test_gpu_paths.py` is not created because there is no GPU path to test.
If one is added later, its test must skip with `pytest.mark.skipif` on a failed
`DeviceContext()` and the README must record the skip. The probes that produced
the numbers above are kept so the decision can be re-checked on a quieter host
or a newer toolchain:

```bash
pixi run python bench/probe_intensity.py      # CPU kernel throughput
pixi run python bench/probe_gpu_floor.py      # CPU kernel time + GPU copy floor
pixi run mojo run bench/probe_gpu.mojo        # device present?
pixi run mojo run bench/probe_gpu_floor.mojo  # GPU copy floor at several sizes
```

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
