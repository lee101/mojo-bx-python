"""Serial and parallel packed-word paths must produce identical results.

The shipped threshold (2**24 words) is far above anything the fast pytest suite
allocates, so the suite would never enter the parallel branch.  This module
therefore builds the same ``src/capi.mojo`` twice - once with the shipped
threshold and once with it forced to 0, so every packed-word loop threads even
at the small sizes below - and compares the two builds on identical buffers.

The branch decision depends on size only, so "threaded at 1536 Mi-bit" exercises
exactly the code that "threaded at 8192 Mi-bit" would.
"""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import numpy as np
import pytest

from bx.bitset import BitSet

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "capi.mojo"
I = ctypes.c_int64
FULL = 0xFFFF_FFFF_FFFF_FFFF
SHIPPED = "comptime PARALLEL_WORD_THRESHOLD = 1 << 24"
FORCED = "comptime PARALLEL_WORD_THRESHOLD = 0"
PARALLEL_TASKS = 4  # src/capi.mojo PARALLEL_TASKS


def _build(threshold_line: str, directory: Path) -> ctypes.CDLL:
    text = SOURCE.read_text()
    assert SHIPPED in text, "src/capi.mojo no longer has the shipped threshold line"
    source = directory / "capi.mojo"
    source.write_text(text.replace(SHIPPED, threshold_line))
    out = directory / "lib.so"
    subprocess.run(["bash", "-c", f"mojo build --emit shared-lib {source} -o {out}"],
                   check=True, capture_output=True)
    lib = ctypes.CDLL(out)
    lib.mbx_bits_set_range.argtypes = [I] * 4
    lib.mbx_bits_count_range.argtypes = [I] * 3
    lib.mbx_bits_count_range.restype = I
    lib.mbx_bits_binary.argtypes = [I] * 4
    lib.mbx_bits_invert.argtypes = [I] * 2
    lib.mbx_bits_next.argtypes = [I] * 4
    return lib


@pytest.fixture(scope="module")
def builds(tmp_path_factory):
    directory = tmp_path_factory.mktemp("builds")
    serial_dir, parallel_dir = directory / "serial", directory / "parallel"
    serial_dir.mkdir()
    parallel_dir.mkdir()
    return (_build(SHIPPED, serial_dir), _build(FORCED, parallel_dir))


# Packed-word sizes in bits.  16 Mi-bit is below the shipped threshold; 192 and
# 1536 Mi-bit are above it, so the production builds also take the threaded path
# where the size permits.  1536 is the largest BitSet the library accepts
# (MAX_INT bits); at ~256 MiB per buffer it is what "a real caller passes" can
# actually reach here.
SIZES = [16 << 20, 192 << 20, 1536 << 20]


@pytest.mark.parametrize("bits", SIZES)
def test_ranged_operations_match_serial(builds, bits):
    serial, parallel = builds
    words = (bits + 63) // 64
    start = 37                      # unaligned first word
    count = (bits - start) // 3 * 2 + 11  # partial last word

    a = np.zeros(words, dtype=np.uint64)
    b = np.zeros(words, dtype=np.uint64)
    ones = np.ones(words, dtype=np.uint64)
    serial.mbx_bits_set_range(a.ctypes.data, start, count, 1)
    parallel.mbx_bits_set_range(b.ctypes.data, start, count, 1)
    np.testing.assert_array_equal(b, a)
    serial.mbx_bits_set_range(a.ctypes.data, start, count, 0)
    parallel.mbx_bits_set_range(b.ctypes.data, start, count, 0)
    np.testing.assert_array_equal(b, a)
    del a, b

    for op in (0, 1, 2):  # iand, ior, ixor
        pattern = np.full(words, 0x0123_4567_89AB_CDEF, dtype=np.uint64)
        a, b = pattern.copy(), pattern.copy()
        serial.mbx_bits_binary(a.ctypes.data, ones.ctypes.data, bits, op)
        parallel.mbx_bits_binary(b.ctypes.data, ones.ctypes.data, bits, op)
        np.testing.assert_array_equal(b, a, err_msg=f"operation {op}")

        # count_range is serial at every size; it must still agree exactly.
        assert (serial.mbx_bits_count_range(a.ctypes.data, 0, bits)
                == parallel.mbx_bits_count_range(b.ctypes.data, 0, bits))
        del pattern, a, b

    pattern = np.full(words, 0x0123_4567_89AB_CDEF, dtype=np.uint64)
    a, b = pattern.copy(), pattern.copy()
    serial.mbx_bits_invert(a.ctypes.data, bits)
    parallel.mbx_bits_invert(b.ctypes.data, bits)
    np.testing.assert_array_equal(b, a)
    if bits % 64:  # no set bit may appear above ``bits`` in the last word
        assert int(a[-1]) < (1 << (bits % 64))
        assert int(b[-1]) < (1 << (bits % 64))


@pytest.mark.parametrize("bits", [192 << 20, 1536 << 20])
def test_public_api_matches_serial_build(builds, bits):
    """Same BitSet operations through the Python API, serial vs threaded build."""
    serial, parallel = builds
    answers = {}
    for label, lib in (("serial", serial), ("threaded", parallel)):
        import bx.bitset as bitset_module

        bitset_module._NATIVE = lib
        try:
            b = BitSet(bits)
            b.set_range(bits // 10, bits // 2)
            other = BitSet(bits)
            other.set_range(0, bits // 3)
            b.iand(other)
            iand_count = b.count_range(0, bits)
            b.invert()
            answers[label] = (iand_count, b.count_range(0, bits), b.next_set(0))
        finally:
            bitset_module._NATIVE = serial
    assert answers["serial"] == answers["threaded"]


def test_worker_cap_present():
    """Guard the contract the rest of the suite assumes."""
    assert PARALLEL_TASKS == 4
    assert f"comptime PARALLEL_TASKS = {PARALLEL_TASKS}" in SOURCE.read_text()
