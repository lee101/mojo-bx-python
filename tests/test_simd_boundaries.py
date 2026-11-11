"""Length boundaries for the native scan/compaction kernels.

The SIMD block width for the int64 kernels is simd_width_of[DType.int64](),
which is 4 on the pinned compiler; the lengths below are derived from the
entry points the Python code calls so the block loop and the scalar tail are
both exercised, including on 16-byte-aligned and unaligned base addresses.
"""

from __future__ import annotations

import numpy as np
import pytest

from bx.align.score import hox70, score_texts
from bx.bitset import BitSet
from bx.intervals.intersection import IntervalTree, overlap_counts

W = 4  # simd_width_of[DType.int64]() on the pinned compiler
LENGTHS = [0, 1, W - 1, W, W + 1, 1003]


def _unaligned(array, offset):
    """A same-content view whose base address is ``offset`` int64s into a buffer."""
    hold = np.zeros(len(array) + offset, dtype=array.dtype)
    hold[offset:] = array
    return hold[offset:]


@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("offset", [0, 1])
def test_overlap_counts_all_lengths(n, offset):
    starts = np.arange(n, dtype=np.int64) * 3
    ends = starts + 11
    qs = np.array([-5, 0, 1, 3, 7, 11, 3 * max(n - 1, 0), 3 * n + 4], dtype=np.int64)
    qe = qs + np.array([3, 1, 2, 7, 40, 1, 1, 100], dtype=np.int64)
    expected = np.fromiter((((ends > a) & (starts < b)).sum() for a, b in zip(qs, qe, strict=True)),
                           dtype=np.int64, count=len(qs))
    got = overlap_counts(_unaligned(starts, offset), _unaligned(ends, offset),
                         _unaligned(qs, offset), _unaligned(qe, offset))
    np.testing.assert_array_equal(got, expected)


def test_overlap_counts_long_and_ragged_not_multiple_of_w():
    n, m = 1003, 257
    rng = np.random.default_rng(7)
    starts = np.sort(rng.integers(0, 5000, size=n))
    ends = starts + rng.integers(1, 60, size=n)
    qs = rng.integers(-10, 5000, size=m)
    qe = qs + rng.integers(0, 900, size=m)
    expected = np.fromiter((((ends > a) & (starts < b)).sum() for a, b in zip(qs, qe, strict=True)),
                           dtype=np.int64, count=m)
    np.testing.assert_array_equal(overlap_counts(starts, ends, qs, qe), expected)


def _bruteforce(items, q_start, q_end):
    return [i for i, (s, e) in enumerate(items) if e > q_start and s < q_end]


@pytest.mark.parametrize("n", LENGTHS)
def test_interval_find_all_lengths(n):
    items = [(i * 3, i * 3 + 1) for i in range(n)] + [(0, 1_000)]
    tree = IntervalTree()
    for index, (start, end) in enumerate(items):
        tree.insert(start, end, index)
    # A query covering everything must return every interval; IntervalTree
    # stores start-sorted values, so compare as a multiset of indices.
    assert sorted(tree.find(-1, 3 * n + 20)) == _bruteforce(items, -1, 3 * n + 20)
    # A query past every interval returns nothing.
    past = max(e for _, e in items) + 1
    assert tree.find(past, past + 100) == []
    # A narrow query inside the long interval; short intervals may straddle it.
    assert sorted(tree.find(500, 501)) == _bruteforce(items, 500, 501)


@pytest.mark.parametrize("n", LENGTHS)
def test_interval_find_matches_bruteforce_for_each_tail_remainder(n):
    rng = np.random.default_rng(n)
    items = []
    for _ in range(n):
        start = int(rng.integers(0, 400))
        items.append((start, start + int(rng.integers(1, 40))))
    tree = IntervalTree()
    for index, (start, end) in enumerate(items):
        tree.insert(start, end, index)
    # widths 0..W and a long odd one; each window length hits a different tail
    for q_start in range(-5, 410, 13):
        for width in (0, 1, 2, W - 1, W, W + 1, 4 * 3 + 1):
            q_end = q_start + width
            assert sorted(tree.find(q_start, q_end)) == _bruteforce(items, q_start, q_end), (
                n, q_start, q_end)


@pytest.mark.parametrize("n", LENGTHS)
def test_bitset_word_loops_all_lengths(n):
    """Packed-word fill/count/invert/boolean loops for every tail remainder."""
    size = 64 * n + 5
    bits = BitSet(size)
    expected = np.zeros(size, dtype=np.uint8)
    bits.set_range(3, max(size - 3, 0))
    expected[3:] = 1
    assert bits.count_range() == int(expected.sum())
    assert bits.next_set(0) == 3

    other = BitSet(size)
    other.set_range(1, 2)
    bits.ior(other)
    expected[1:3] = 1
    assert bits.count_range() == int(expected.sum())

    bits.iand(other)
    only_other = np.zeros(size, dtype=np.uint8)
    only_other[1:3] = 1
    expected = np.logical_and(expected, only_other).astype(np.uint8)
    assert bits.count_range() == int(expected.sum())

    ~bits
    assert bits.count_range() == int((1 - expected).sum())
    assert bits.next_set(0) == int(np.flatnonzero(1 - expected)[0])


@pytest.mark.parametrize("n", LENGTHS)
def test_score_texts_all_lengths_exact(n):
    """Gap state carried through the whole length, block boundary and tail."""
    left = "ACGT-AC-T"
    right = "AG-T-AG-T"
    text1 = (left * (n // len(left) + 1))[:n]
    text2 = (right * (n // len(right) + 1))[:n]

    def reference(a, b):
        total = 0.0
        last_gap_a = last_gap_b = False
        for ca, cb in zip(a, b, strict=True):
            if ca == "-" and cb == "-":
                continue
            if ca == "-":
                total -= hox70.gap_extend
                if not last_gap_a:
                    total -= hox70.gap_open
                    last_gap_a, last_gap_b = True, False
            elif cb == "-":
                total -= hox70.gap_extend
                if not last_gap_b:
                    total -= hox70.gap_open
                    last_gap_a, last_gap_b = False, True
            else:
                total += hox70.table[ord(ca), ord(cb)]
                last_gap_a = last_gap_b = False
        return total

    assert score_texts(hox70, text1, text2) == pytest.approx(reference(text1, text2), rel=0, abs=1e-9)
