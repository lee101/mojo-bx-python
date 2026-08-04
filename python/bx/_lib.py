"""Shared-library loading and ndarray helpers used by the public modules."""

from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LIBRARY = ROOT / "dist" / "libmojo-bx-python.so"
I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mbx_find_indices": ([I, I, I, I, I, I], I),
    "mbx_find_bounds": ([I, I, I, I, I], I),
    "mbx_find_indices_window": ([I, I, I, I, I], I),
    "mbx_overlap_counts": ([I] * 7, None),
    "mbx_bits_set_range": ([I] * 4, None),
    "mbx_bits_count_range": ([I] * 3, I),
    "mbx_bits_next": ([I] * 4, I),
    "mbx_bits_binary": ([I] * 4, None),
    "mbx_bits_invert": ([I] * 2, None),
    "mbx_score_texts": ([I, I, I, I, I, I, I, F, F], F),
    "mbx_accumulate_scores": ([I, I, I, I, I, I, I, F, F, I, I], I),
}

_library: ctypes.CDLL | None = None


def build() -> Path:
    """Build the ABI library when it is missing or older than its source."""
    source = ROOT / "src" / "capi.mojo"
    if not LIBRARY.exists() or LIBRARY.stat().st_mtime < source.stat().st_mtime:
        subprocess.run(["bash", str(ROOT / "build" / "build.sh")], check=True, cwd=ROOT)
    return LIBRARY


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def addr(array: np.ndarray) -> int:
    return int(array.ctypes.data)


def i64(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.int64)


def u8(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.uint8)
