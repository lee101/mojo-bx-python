"""Mojo implementations of bx-python's mutable bit-vector API."""

from __future__ import annotations

import numpy as np

from ._lib import addr, lib

MAX_INT = 2_147_483_647
MAX = 512 * 1024 * 1024


class BitSet:
    def __init__(self, bitCount):
        if bitCount > MAX_INT:
            raise ValueError(f"{bitCount} is larger than the maximum BitSet size of {MAX_INT}.")
        if bitCount < 0:
            raise ValueError("BitSet size must be non-negative.")
        self.bitCount = int(bitCount)
        self._bits = np.zeros(self.bitCount, dtype=np.uint8)

    @property
    def size(self):
        return self.bitCount

    def _index(self, index):
        if index < 0:
            raise IndexError(f"BitSet index ({index}) must be non-negative.")
        if index >= self.bitCount:
            raise IndexError(f"{index} is larger than the size of this BitSet ({self.bitCount}).")

    def _range_count(self, start, count):
        self._index(start)
        if count < 0:
            raise IndexError(f"Count ({count}) must be non-negative.")
        if start + count > self.bitCount:
            raise IndexError(f"End {start + count} is larger than the size of this BitSet ({self.bitCount}).")

    def _same_size(self, other):
        if self.bitCount != other.bitCount:
            raise ValueError("BitSets must have the same size")

    def set(self, index):
        self._index(index)
        self._bits[index] = 1

    def clear(self, index):
        self._index(index)
        self._bits[index] = 0

    def clone(self):
        other = type(self)(self.bitCount)
        other._bits[:] = self._bits
        return other

    def set_range(self, start, count):
        self._range_count(start, count)
        lib().mbx_bits_set_range(addr(self._bits), start, count, 1)

    def get(self, index):
        self._index(index)
        return int(self._bits[index])

    def count_range(self, start=0, count=None):
        if count is None:
            count = self.bitCount - start
        self._range_count(start, count)
        return int(lib().mbx_bits_count_range(addr(self._bits), start, count))

    def _next(self, start, end, value):
        self._index(start)
        if end is None:
            end = self.bitCount
        if end < start:
            raise IndexError(f"Range end ({end}) must be greater than range start({start}).")
        if end > self.bitCount:
            raise IndexError(f"End {end} is larger than the size of this BitSet ({self.bitCount}).")
        found = int(lib().mbx_bits_next(addr(self._bits), start, end, value))
        return found

    def next_set(self, start, end=None):
        return self._next(start, end, 1)

    def next_clear(self, start, end=None):
        return self._next(start, end, 0)

    def iand(self, other):
        self._same_size(other)
        lib().mbx_bits_binary(addr(self._bits), addr(other._bits), self.bitCount, 0)

    def ior(self, other):
        self._same_size(other)
        lib().mbx_bits_binary(addr(self._bits), addr(other._bits), self.bitCount, 1)

    def ixor(self, other):
        self._same_size(other)
        lib().mbx_bits_binary(addr(self._bits), addr(other._bits), self.bitCount, 2)

    def invert(self):
        lib().mbx_bits_invert(addr(self._bits), self.bitCount)

    def __getitem__(self, index):
        return self.get(index)

    def __iand__(self, other):
        self.iand(other)
        return self

    def __ior__(self, other):
        self.ior(other)
        return self

    def __invert__(self):
        self.invert()
        return self


class BinnedBitSet(BitSet):
    """API-compatible dense implementation of bx-python's binned bit set."""

    def __init__(self, size=MAX, granularity=1024):
        super().__init__(size)
        self._bin_size = int(granularity)

    @property
    def bin_size(self):
        return self._bin_size

    def next_set(self, start):
        return super().next_set(start)

    def next_clear(self, start):
        return super().next_clear(start)
