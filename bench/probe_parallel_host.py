"""Run the ctypes-host parallelize probe (lib must already be built)."""
import ctypes, sys
import numpy as np

lib = ctypes.CDLL(sys.argv[1])
for name in ("probe_parallel_fill", "probe_parallel_fill_2arg"):
    fn = getattr(lib, name); fn.argtypes = [ctypes.c_int64, ctypes.c_int64]; fn.restype = ctypes.c_int64
n = 4000
for name in ("probe_parallel_fill", "probe_parallel_fill_2arg"):
    buf = np.full(n, -1, dtype=np.int64)
    got = getattr(lib, name)(int(buf.ctypes.data), n)
    print(name, "sum_ok=", got == (np.arange(n) * 2).sum(), "all_written=", bool((buf == np.arange(n) * 2).all()), flush=True)
