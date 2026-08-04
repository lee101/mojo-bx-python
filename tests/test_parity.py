from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from bx.align.score import (ScoringScheme, accumulate_scores, build_scoring_scheme,
                            hox70, read_scoring_scheme, score_alignment, score_texts)
from bx.bitset import BinnedBitSet, BitSet
from bx.intervals.intersection import Interval, IntervalNode, IntervalTree, overlap_counts


def upstream(script):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    with tempfile.TemporaryDirectory() as directory:
        result = subprocess.run([sys.executable, "-c", script], cwd=directory, env=env,
                                check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


def test_interval_find_and_neighbors_match_upstream():
    tree = IntervalTree()
    intervals = [(5, 12), (1, 4), (8, 10), (15, 21), (22, 26)]
    for start, end in intervals:
        tree.insert(start, end, Interval(start, end))
    actual = {"find": [x.start for x in tree.find(9, 20)], "before": [x.start for x in tree.before(15, 3, 20)], "after": [x.start for x in tree.after(8, 3, 20)]}
    expected = upstream("""
import json
from bx.intervals.intersection import IntervalTree
t = IntervalTree()
for x in [(5,12),(1,4),(8,10),(15,21),(22,26)]: t.insert(x[0], x[1], type('I', (), {'start':x[0], 'end':x[1]})())
print(json.dumps({'find': [x.start for x in t.find(9,20)], 'before': [x.start for x in t.before(15,3,20)], 'after': [x.start for x in t.after(8,3,20)]}))
""")
    assert actual["find"] == expected["find"]
    assert sorted(actual["before"]) == sorted(expected["before"])
    assert actual["after"] == expected["after"]


def test_interval_find_keeps_long_early_intervals():
    tree = IntervalTree()
    tree.insert(0, 1_000, "long")
    tree.insert(100, 110, "short")
    tree.insert(200, 210, "later")
    assert tree.find(500, 501) == ["long"]


def test_interval_object_shortcuts_and_strands():
    tree = IntervalTree()
    for start in range(0, 100, 10):
        tree.add_interval(Interval(start, start + 9))
    assert [x.start for x in tree.before_interval(Interval(50, 60), 2)] == [40, 30]
    assert [x.start for x in tree.after_interval(Interval(50, 60), 2)] == [70, 80]
    assert [x.start for x in tree.upstream_of_interval(Interval(50, 60, strand="-"), 2)] == [70, 80]
    assert tree.find(9, 10) == []
    assert [x.start for x in tree.downstream_of_interval(Interval(50, 60, strand="-"), 2)] == [40, 30]
    seen = []
    tree.traverse(lambda node: seen.append((node.start, node.end)))
    assert seen == [(start, start + 9) for start in range(0, 100, 10)]


def test_interval_node_and_bulk_counts():
    node = IntervalNode(10, 20, "one")
    node = node.insert(3, 7, "two").insert(21, 23, "three")
    assert node.find(6, 22) == ["two", "one", "three"]
    assert node.left(21, n=2) == ["one", "two"]
    assert node.right(7, n=2) == ["one", "three"]
    np.testing.assert_array_equal(overlap_counts([20, 0, 8], [25, 10, 12], [1, 9, 12], [2, 10, 22]), [1, 2, 1])


@pytest.mark.parametrize("kind", [BitSet, BinnedBitSet])
def test_bitsets_match_upstream(kind):
    bits = kind(100, 10) if kind is BinnedBitSet else kind(100)
    bits.set_range(11, 3)
    bits.set_range(20, 55)
    bits.clear(21)
    actual = {"count": bits.count_range(0, 100), "set": bits.next_set(15), "clear": bits.next_clear(20), "values": [bits[i] for i in range(100)]}
    expected = upstream("""
import json
from bx.bitset import BinnedBitSet
b=BinnedBitSet(100,10); b.set_range(11,3); b.set_range(20,55); b.clear(21)
print(json.dumps({'count':b.count_range(0,100),'set':b.next_set(15),'clear':b.next_clear(20),'values':[b[i] for i in range(100)]}))
""")
    assert actual == expected


def test_bitset_boolean_operations_and_bounds():
    first, second = BitSet(20), BitSet(20)
    first.set_range(2, 9)
    second.set_range(7, 8)
    first.iand(second)
    assert first.count_range() == 4
    first.ior(second)
    assert first.count_range() == 8
    first.ixor(second)
    assert first.count_range() == 0
    ~first
    assert first.count_range() == 20
    with pytest.raises(IndexError):
        first.set_range(20, 0)
    with pytest.raises(ValueError):
        first.iand(BitSet(19))


def test_bitset_simd_tail_and_unaligned_range():
    bits = BitSet(3 + 2 * 67)
    bits.set_range(3, 67)
    assert bits.count_range(3, 67) == 67
    assert bits.count_range(0, bits.size) == 67
    ~bits
    assert bits.count_range(3, 67) == 0
    assert bits.count_range(70, 67) == 67
    other = BitSet(bits.size)
    other.set_range(3, 67)
    bits.ior(other)
    assert bits.count_range(3, 67) == 67
    bits.iand(other)
    assert bits.count_range(3, 67) == 67
    bits.ixor(other)
    assert bits.count_range() == 0


def test_alignment_scores_match_published_upstream_vectors():
    pairs = [
        ("CCACTAGTTTTTAAATAATCTACTATCAAATAAAAGATTTGTTAATAATAAATTTTAAATCATTAACACTT", "CCATTTGGGTTCAAAAATTGATCTATCA----------TGGTGGATTATTATTTAGCCATTAAGGACAAAT", -111),
        ("CCACTAGTTTTTGATTC", "CCATTTGGGTTC-----", -299),
        ("CTTAGTTTTTGATCACC", "-----CTTGGGTTTACC", -299),
    ]
    for left, right, expected in pairs:
        assert score_texts(hox70, left, right) == expected
    np.testing.assert_allclose(accumulate_scores(hox70, "-----CTTT", "CTTAGTTTA"), [-430, -460, -490, -520, -550, -581, -490, -399, -522])
    np.testing.assert_allclose(accumulate_scores(hox70, "-----CTTT", "CTTAGTTTA", skip_ref_gaps=True), [-581, -490, -399, -522])


def test_alignment_matrix_builder_matches_upstream():
    matrix = """  A C G T
       2 -1 -1 -1
      -1  2 -1 -1
      -1 -1  2 -1
      -1 -1 -1  2"""
    ours = build_scoring_scheme(matrix, 5, 1)
    expected = upstream("""
import json
from bx.align.score import build_scoring_scheme, score_texts
s=build_scoring_scheme('  A C G T\\n 2 -1 -1 -1\\n-1 2 -1 -1\\n-1 -1 2 -1\\n-1 -1 -1 2', 5, 1)
print(json.dumps([int(score_texts(s, 'AC-GT', 'ATCG-')), s.table.tolist()]))
""")
    assert score_texts(ours, "AC-GT", "ATCG-") == expected[0]
    assert ours.table.tolist() == expected[1]


def test_alignment_reader_score_alignment_and_safe_fallbacks(tmp_path):
    matrix = "  A C\n 2 -1\n-1  2\n"
    path = tmp_path / "matrix.txt"
    path.write_text(matrix)
    scheme = read_scoring_scheme(path, 5, 1)
    assert score_texts(scheme, "AC", "AC") == 4

    class Alignment:
        components = [type("Component", (), {"text": "AC"})(), type("Component", (), {"text": "AC"})()]

    assert score_alignment(scheme, Alignment()) == 4
    # Out-of-table and non-Latin-1 characters must use checked Python indexing
    # instead of becoming unchecked byte/table offsets in Mojo.
    with pytest.raises(IndexError):
        score_texts(hox70, "é", "é")
    with pytest.raises(IndexError):
        score_texts(hox70, "Ā", "Ā")
    precise = ScoringScheme(0, 0, text1_range=1, text2_range=1, typecode=np.int64)
    precise.table[0, 0] = 2**53 + 1
    assert score_texts(precise, "\0", "\0") == 2**53 + 1
