"""Half-open interval queries with bx-python compatible objects and methods."""

from __future__ import annotations

import numpy as np

from bx._lib import addr, i64, lib


class Interval:
    def __init__(self, start, end, value=None, chrom=None, strand=None):
        assert start <= end, "start must be less than end"
        self.start = int(start)
        self.end = int(end)
        self.value = value
        self.chrom = chrom
        self.strand = strand

    def __repr__(self):
        text = f"Interval({self.start}, {self.end}"
        if self.value is not None:
            text += ", value=" + str(self.value)
        return text + ")"

    def __eq__(self, other):
        return isinstance(other, Interval) and self.start == other.start and self.end == other.end

    def __lt__(self, other):
        return self.start < other.start or self.end < other.end

    def __le__(self, other):
        return self == other or self < other

    def __gt__(self, other):
        return self.start > other.start or self.end > other.end

    def __ge__(self, other):
        return self == other or self > other


class IntervalNode:
    """Compatibility node backed by a sorted collection rather than a treap."""

    def __init__(self, start, end, interval):
        self.start, self.end, self.interval = int(start), int(end), interval
        self._items = [(self.start, self.end, interval)]

    @property
    def left_node(self):
        return None

    @property
    def right_node(self):
        return None

    @property
    def root_node(self):
        return None

    def __repr__(self):
        return f"IntervalNode({self.start}, {self.end})"

    def insert(self, start, end, interval):
        self._items.append((int(start), int(end), interval))
        self._items.sort(key=lambda x: (x[0], x[1]))
        return self

    def intersect(self, start, end, sort=True):
        return [value for s, e, value in self._items if e > start and s < end]

    find = intersect

    def left(self, position, n=1, max_dist=2500):
        result = [x for x in self._items if x[1] < position and position - x[1] <= max_dist]
        return [x[2] for x in sorted(result, key=lambda x: x[1], reverse=True)[:n]]

    def right(self, position, n=1, max_dist=2500):
        result = [x for x in self._items if x[0] > position and x[0] - position <= max_dist]
        return [x[2] for x in sorted(result, key=lambda x: x[0])[:n]]

    def traverse(self, func):
        for start, end, value in self._items:
            node = IntervalNode(start, end, value)
            func(node)


class IntervalTree:
    def __init__(self):
        self._items = []
        self._dirty = True

    def _arrays(self):
        if self._dirty:
            self._items.sort(key=lambda x: (x[0], x[1]))
            self._starts = i64([x[0] for x in self._items])
            self._ends = i64([x[1] for x in self._items])
            self._max_ends = np.maximum.accumulate(self._ends)
            self._values = [x[2] for x in self._items]
            self._dirty = False
        return self._starts, self._ends, self._max_ends

    def insert(self, start, end, value=None):
        self._items.append((int(start), int(end), value))
        self._dirty = True

    add = insert

    def find(self, start, end):
        if not self._items:
            return []
        starts, ends, max_ends = self._arrays()
        packed = int(lib().mbx_find_bounds(addr(starts), addr(max_ends), len(starts), int(start), int(end)))
        base = len(starts) + 1
        lo, hi = packed // base, packed % base
        indices = np.empty(hi - lo, dtype=np.int64)
        count = lib().mbx_find_indices_window(addr(ends), lo, hi, int(start), addr(indices))
        return [self._values[i] for i in indices[:count].tolist()]

    def before(self, position, num_intervals=1, max_dist=2500):
        candidates = [x for x in self._items if x[1] < position and position - x[1] <= max_dist]
        return [x[2] for x in sorted(candidates, key=lambda x: x[1], reverse=True)[:num_intervals]]

    def after(self, position, num_intervals=1, max_dist=2500):
        candidates = [x for x in self._items if x[0] > position and x[0] - position <= max_dist]
        return [x[2] for x in sorted(candidates, key=lambda x: x[0])[:num_intervals]]

    def insert_interval(self, interval):
        self.insert(interval.start, interval.end, interval)

    add_interval = insert_interval

    def before_interval(self, interval, num_intervals=1, max_dist=2500):
        return self.before(interval.start, num_intervals, max_dist)

    def after_interval(self, interval, num_intervals=1, max_dist=2500):
        return self.after(interval.end, num_intervals, max_dist)

    def upstream_of_interval(self, interval, num_intervals=1, max_dist=2500):
        if interval.strand == -1 or interval.strand == "-":
            return self.after(interval.end, num_intervals, max_dist)
        return self.before(interval.start, num_intervals, max_dist)

    def downstream_of_interval(self, interval, num_intervals=1, max_dist=2500):
        if interval.strand == -1 or interval.strand == "-":
            return self.before(interval.start, num_intervals, max_dist)
        return self.after(interval.end, num_intervals, max_dist)

    def traverse(self, fn):
        if not self._items:
            return None
        for start, end, value in sorted(self._items, key=lambda x: (x[0], x[1])):
            fn(IntervalNode(start, end, value))


Intersecter = IntervalTree


def overlap_counts(starts, ends, query_starts, query_ends):
    """Return counts of half-open interval overlaps for each query range."""
    starts, ends = i64(starts), i64(ends)
    query_starts, query_ends = i64(query_starts), i64(query_ends)
    if len(starts) != len(ends) or len(query_starts) != len(query_ends):
        raise ValueError("start and end arrays must have matching lengths")
    order = np.argsort(starts, kind="stable")
    starts, ends = starts[order], ends[order]
    result = np.empty(len(query_starts), dtype=np.int64)
    lib().mbx_overlap_counts(addr(starts), addr(ends), len(starts), addr(query_starts), addr(query_ends), len(query_starts), addr(result))
    return result
