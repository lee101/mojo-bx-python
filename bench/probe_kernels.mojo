"""Experimental per-worker entry points for the thread-scaling probe.

Not part of the shipped ABI.  ``threads == 0`` is the serial path; otherwise
the work is split into ``threads`` contiguous chunks launched with
``parallelize``.  Used only by bench/thread_scale.py to locate the threshold.
"""

from std.bit import pop_count
from std.memory import alloc
from max.algorithm import parallelize

comptime W = 4
comptime U64Ptr = Pointer[UInt64, AnyOrigin[mut=True]]


def _p(addr: Int) -> U64Ptr:
    return U64Ptr(unsafe_from_address=addr)


def _fill(p: U64Ptr, count: Int, value: UInt64):
    var vector_end = count - count % W
    var values = SIMD[DType.uint64, W](value)
    for i in range(0, vector_end, W):
        p.store(i, values)
    for i in range(vector_end, count):
        p.unsafe_store(i, value)


def _binary(a: U64Ptr, b: U64Ptr, count: Int, op: Int):
    var vector_end = count - count % W
    for i in range(0, vector_end, W):
        var l = a.unsafe_load[width=W](i)
        var r = b.unsafe_load[width=W](i)
        if op == 0:
            a.store(i, l & r)
        elif op == 1:
            a.store(i, l | r)
        else:
            a.store(i, l ^ r)
    for i in range(vector_end, count):
        if op == 0:
            a.unsafe_store(i, a.unsafe_load(i) & b.unsafe_load(i))
        elif op == 1:
            a.unsafe_store(i, a.unsafe_load(i) | b.unsafe_load(i))
        else:
            a.unsafe_store(i, a.unsafe_load(i) ^ b.unsafe_load(i))


def _popcount(p: U64Ptr, count: Int) -> Int:
    var total = 0
    var vector_end = count - count % W
    for i in range(0, vector_end, W):
        total += Int(pop_count(p.unsafe_load[width=W](i)).reduce_add())
    for i in range(vector_end, count):
        total += Int(pop_count(p.unsafe_load(i)))
    return total


def _invert(p: U64Ptr, count: Int):
    var vector_end = count - count % W
    for i in range(0, vector_end, W):
        p.store(i, ~p.unsafe_load[width=W](i))
    for i in range(vector_end, count):
        p.unsafe_store(i, ~p.unsafe_load(i))


def _chunks(count: Int, threads: Int) -> Int:
    var chunk = (count + threads - 1) // threads
    return (chunk + W - 1) // W * W


@export("mbx_probe_set_range")
def mbx_probe_set_range(bits: Int, start: Int, count: Int, value: Int, threads: Int) abi("C"):
    var b = _p(bits)
    var words = (start + count + 63) // 64
    var v = ~UInt64(0) if value != 0 else UInt64(0)
    if threads <= 1:
        _fill(b, words, v)
        return
    var chunk = _chunks(words, threads)
    def work(k: Int) {imm}:
        var lo = k * chunk
        var hi = min(lo + chunk, words)
        if lo < hi:
            _fill(b.unsafe_offset(lo), hi - lo, v)
    parallelize(work, threads, threads)


@export("mbx_probe_count_range")
def mbx_probe_count_range(bits: Int, start: Int, count: Int, threads: Int) abi("C") -> Int:
    var b = _p(bits)
    var words = (start + count + 63) // 64
    if threads <= 1:
        return _popcount(b, words)
    var chunk = _chunks(words, threads)
    var partial = alloc[Int](threads)
    try:
        def work(k: Int) {imm}:
            var lo = k * chunk
            var hi = min(lo + chunk, words)
            partial.unsafe_store(k, _popcount(b.unsafe_offset(lo), hi - lo) if lo < hi else 0)
        parallelize(work, threads, threads)
        var total = 0
        for k in range(threads):
            total += Int(partial.unsafe_load(k))
        partial.free()
        return total
    except:
        partial.free()
        return -1


@export("mbx_probe_binary")
def mbx_probe_binary(bits: Int, other: Int, size: Int, operation: Int, threads: Int) abi("C"):
    var a = _p(bits)
    var b = _p(other)
    var words = (size + 63) // 64
    if threads <= 1:
        _binary(a, b, words, operation)
        return
    var chunk = _chunks(words, threads)
    def work(k: Int) {imm}:
        var lo = k * chunk
        var hi = min(lo + chunk, words)
        if lo < hi:
            _binary(a.unsafe_offset(lo), b.unsafe_offset(lo), hi - lo, operation)
    parallelize(work, threads, threads)


@export("mbx_probe_invert")
def mbx_probe_invert(bits: Int, size: Int, threads: Int) abi("C"):
    var b = _p(bits)
    var words = (size + 63) // 64
    if threads <= 1:
        _invert(b, words)
        return
    var chunk = _chunks(words, threads)
    def work(k: Int) {imm}:
        var lo = k * chunk
        var hi = min(lo + chunk, words)
        if lo < hi:
            _invert(b.unsafe_offset(lo), hi - lo)
    parallelize(work, threads, threads)
