"""C ABI kernels for the covered bx-python subset.

The bindings own every buffer.  Addresses are passed as Int because exported
Mojo functions cannot carry a pointer origin in their public signature.
"""

from std.bit import count_trailing_zeros, pop_count
from std.sys import simd_width_of as simdwidthof
from max.algorithm import parallelize

comptime U8Ptr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime U64Ptr = UnsafePointer[UInt64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int, AnyOrigin[mut=True]]
comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime W = simdwidthof[DType.float64]()
# Packed-word loops run on worker threads only above this many words (2**24
# words = 128 Mi-bit = 16 MiB of packed storage).  Measured crossover and the
# margin above it are in README "Parallelism".  Below it the serial loop wins:
# a threaded launch costs ~2-7 ms of fixed wait/spawn/rejoin on this host, and
# the serial loop touches 16 MiB in well under 1 ms, so a smaller cutoff only
# ever pays the launch.
comptime PARALLEL_WORD_THRESHOLD = 1 << 24
# The host is shared; 4 workers is the cap.  Measured scaling flattens past it.
comptime PARALLEL_TASKS = 4


def u8(addr: Int) -> U8Ptr:
    return U8Ptr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def u64(addr: Int) -> U64Ptr:
    return U64Ptr(unsafe_from_address=addr)


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


@export("mbx_find_indices")
def mbx_find_indices(starts: Int, ends: Int, n: Int, start: Int, end: Int, dst: Int) abi("C") -> Int:
    """Write sorted interval indices overlapping [start, end), return count."""
    var count = 0
    var s = ip(starts)
    var e = ip(ends)
    var result = ip(dst)
    for i in range(n):
        if s[i] >= end:
            break
        if e[i] > start:
            result[count] = i
            count += 1
    return count


def lower_bound(values: IPtr, n: Int, needle: Int) -> Int:
    var lo = 0
    var hi = n
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        if values[mid] < needle:
            lo = mid + 1
        else:
            hi = mid
    return lo


def upper_bound(values: IPtr, n: Int, needle: Int) -> Int:
    var lo = 0
    var hi = n
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        if values[mid] <= needle:
            lo = mid + 1
        else:
            hi = mid
    return lo


@export("mbx_find_bounds")
def mbx_find_bounds(starts: Int, max_ends: Int, n: Int, start: Int, end: Int) abi("C") -> Int:
    """Pack the candidate window for an overlap query into one Int."""
    var lo = upper_bound(ip(max_ends), n, start)
    var hi = lower_bound(ip(starts), n, end)
    return lo * (n + 1) + hi


@export("mbx_find_indices_window")
def mbx_find_indices_window(ends: Int, lo: Int, hi: Int, start: Int, dst: Int) abi("C") -> Int:
    """Filter a previously bounded interval window and write absolute indices."""
    var e = ip(ends)
    var result = ip(dst)
    var count = 0
    for i in range(lo, hi):
        if e[i] > start:
            result[count] = i
            count += 1
    return count


@export("mbx_find_indices_bounded")
def mbx_find_indices_bounded(starts: Int, ends: Int, max_ends: Int, n: Int, start: Int, end: Int, dst: Int) abi("C") -> Int:
    """Bound and filter an overlap query in one FFI call."""
    var lo = upper_bound(ip(max_ends), n, start)
    var hi = lower_bound(ip(starts), n, end)
    if lo >= hi:
        return 0
    return filter_indices(ip(ends), lo, hi, start, ip(dst))


def filter_indices(e: IPtr, lo: Int, hi: Int, start: Int, dst: IPtr) -> Int:
    """Write indices in [lo, hi) whose end is after ``start``, compacted.

    Iteration order is part of the result, but the lanes in a block are
    independent, so the compare runs on a SIMD block and only the lanes that
    hit are stored.  Isolated on the real caller window size (170 candidates,
    2000 windows): scalar 0.32-0.34 ms, this form 0.14-0.15 ms.  End to end
    that change sits inside the +-3% noise of IntervalTree.find, which is
    dominated by the per-query Python and ctypes cost.
    """
    comptime I = SIMD[DType.int64, W]
    var needle = I(start)
    var count = 0
    var i = lo
    while i + W <= hi:
        var hits = e.unsafe_load[width=W](i).cast[DType.int64]().gt(needle).cast[DType.int64]()
        if Int(hits.reduce_add()) != 0:
            for k in range(W):
                if hits[k] != 0:
                    dst.unsafe_store(count, i + k)
                    count += 1
        i += W
    while i < hi:
        if e.unsafe_load(i) > start:
            dst.unsafe_store(count, i)
            count += 1
        i += 1
    return count


@export("mbx_overlap_counts")
def mbx_overlap_counts(starts: Int, ends: Int, max_ends: Int, n: Int, qstarts: Int, qends: Int, m: Int, dst: Int) abi("C"):
    """Count overlaps for many queries against start-sorted intervals.

    ``starts`` is ascending and ``max_ends`` is its running maximum, so both
    ends of the candidate window are binary searches instead of a scan that
    breaks early.  Once the prefix whose running max end is at or below the
    query start is skipped, every remaining iteration is independent and the
    comparison is a SIMD reduction with a scalar tail.
    """
    var s = ip(starts)
    var e = ip(ends)
    var me = ip(max_ends)
    var qs = ip(qstarts)
    var qe = ip(qends)
    var result = ip(dst)
    comptime I = SIMD[DType.int64, W]
    for q in range(m):
        var hi = lower_bound(s, n, qe.unsafe_load(q))
        var lo = upper_bound(me, n, qs.unsafe_load(q))
        if lo >= hi:
            result[q] = 0
            continue
        var needle = I(qs.unsafe_load(q))
        var count = 0
        var i = lo
        while i + W <= hi:
            var hits = e.unsafe_load[width=W](i).cast[DType.int64]().gt(needle).cast[DType.int64]()
            count += Int(hits.reduce_add())
            i += W
        while i < hi:
            if e.unsafe_load(i) > qs.unsafe_load(q):
                count += 1
            i += 1
        result[q] = count


def low_mask(bits: Int) -> UInt64:
    if bits >= 64:
        return ~UInt64(0)
    return (UInt64(1) << UInt64(bits)) - UInt64(1)


def words_chunk(count: Int, parts: Int) -> Int:
    """Words per unit, rounded up to a whole SIMD block so lanes never straddle."""
    var chunk = (count + parts - 1) // parts
    return (chunk + W - 1) // W * W


def fill_words_hw(b: U64Ptr, start: Int, count: Int, value: UInt64):
    """Fill with 256-bit stores: 4 W-wide stores per 64-byte cache line."""
    comptime H = 4
    var values = SIMD[DType.uint64, W](value)
    var i = 0
    while i + H * W <= count:
        var j = 0
        while j < H:
            b.store(start + i + j * W, values)
            j += 1
        i += H * W
    while i + W <= count:
        b.store(start + i, values)
        i += W
    while i < count:
        b.unsafe_store(start + i, value)
        i += 1


@export("mbx_bits_set_range")
def mbx_bits_set_range(bits: Int, start: Int, count: Int, value: Int) abi("C"):
    if count == 0:
        return
    var b = u64(bits)
    var first = start // 64
    var last = (start + count - 1) // 64
    var first_offset = start % 64
    var end_offset = (start + count) % 64
    if first == last:
        var mask = low_mask(count) << UInt64(first_offset)
        if value != 0:
            b[first] |= mask
        else:
            b[first] &= ~mask
        return
    var first_mask = ~UInt64(0) << UInt64(first_offset)
    var last_mask = low_mask(end_offset if end_offset != 0 else 64)
    if value != 0:
        b[first] |= first_mask
        b[last] |= last_mask
    else:
        b[first] &= ~first_mask
        b[last] &= ~last_mask
    var middle_start = first + 1
    var middle_count = last - middle_start
    var fill = ~UInt64(0) if value != 0 else UInt64(0)
    # One wide body for both paths: the 256-bit fill wins below the threshold
    # and inside a worker, so there is no shape that beats it at these sizes.
    if middle_count >= PARALLEL_WORD_THRESHOLD:
        var chunk = words_chunk(middle_count, PARALLEL_TASKS)
        def work(task: Int) {imm}:
            var offset = task * chunk
            var stop = min(offset + chunk, middle_count)
            if offset < stop:
                fill_words_hw(b.unsafe_offset(middle_start + offset), 0, stop - offset, fill)
        parallelize(work, PARALLEL_TASKS, PARALLEL_TASKS)
    else:
        fill_words_hw(b, middle_start, middle_count, fill)


def count_words(b: U64Ptr, start: Int, count: Int) -> Int:
    var total = 0
    var vector_end = count - count % W
    for i in range(0, vector_end, W):
        total += Int(pop_count(b.unsafe_load[width=W](start + i)).reduce_add())
    for i in range(vector_end, count):
        total += Int(pop_count(b[start + i]))
    return total


@export("mbx_bits_count_range")
def mbx_bits_count_range(bits: Int, start: Int, count: Int) abi("C") -> Int:
    """Count set bits in [start, start+count).

    Deliberately serial at every size.  The counting read itself threads well
    (3.5-3.7x on 4 workers at 4 GiB), but the variants that keep each worker's
    data in cache while counting it all lost at the sizes a caller reaches
    (README "Parallelism"), and a threaded scan costs ~6 ms of launch, which the
    caller-visible range cannot absorb.
    """
    if count == 0:
        return 0
    var b = u64(bits)
    var total = 0
    var first = start // 64
    var last = (start + count - 1) // 64
    var first_offset = start % 64
    var end_offset = (start + count) % 64
    if first == last:
        return Int(pop_count(b[first] & (low_mask(count) << UInt64(first_offset))))
    total += Int(pop_count(b[first] & (~UInt64(0) << UInt64(first_offset))))
    total += Int(pop_count(b[last] & low_mask(end_offset if end_offset != 0 else 64)))
    var middle_start = first + 1
    var middle_count = last - middle_start
    total += count_words(b, middle_start, middle_count)
    return total


@export("mbx_bits_next")
def mbx_bits_next(bits: Int, start: Int, size: Int, value: Int) abi("C") -> Int:
    var b = u64(bits)
    var first = start // 64
    var last = (size - 1) // 64
    for word in range(first, last + 1):
        var candidate = b[word] if value != 0 else ~b[word]
        if word == first:
            candidate &= ~UInt64(0) << UInt64(start % 64)
        if word == last and size % 64 != 0:
            candidate &= low_mask(size % 64)
        if candidate != 0:
            return word * 64 + Int(count_trailing_zeros(candidate))
    return size


@export("mbx_bits_binary")
def mbx_bits_binary(bits: Int, other: Int, size: Int, operation: Int) abi("C"):
    var a = u64(bits)
    var b = u64(other)
    var words = (size + 63) // 64
    if words >= PARALLEL_WORD_THRESHOLD:
        var chunk = words_chunk(words, PARALLEL_TASKS)
        def work(task: Int) {imm}:
            var offset = task * chunk
            var stop = min(offset + chunk, words)
            if offset < stop:
                bits_binary_words(a.unsafe_offset(offset), b.unsafe_offset(offset), 0, stop - offset, operation)
        parallelize(work, PARALLEL_TASKS, PARALLEL_TASKS)
    else:
        bits_binary_words(a, b, 0, words, operation)


def bits_binary_words(a: U64Ptr, b: U64Ptr, start: Int, count: Int, operation: Int):
    var vector_end = count - count % W
    for i in range(0, vector_end, W):
        var offset = start + i
        var left = a.unsafe_load[width=W](offset)
        var right = b.unsafe_load[width=W](offset)
        if operation == 0:
            a.store(offset, left & right)
        elif operation == 1:
            a.store(offset, left | right)
        else:
            a.store(offset, left ^ right)
    for i in range(vector_end, count):
        var offset = start + i
        if operation == 0:
            a[offset] = a[offset] & b[offset]
        elif operation == 1:
            a[offset] = a[offset] | b[offset]
        else:
            a[offset] = a[offset] ^ b[offset]


def invert_words(b: U64Ptr, start: Int, count: Int):
    var vector_end = count - count % W
    for i in range(0, vector_end, W):
        var offset = start + i
        b.store(offset, ~b.unsafe_load[width=W](offset))
    for i in range(vector_end, count):
        b[start + i] = ~b[start + i]


@export("mbx_bits_invert")
def mbx_bits_invert(bits: Int, size: Int) abi("C"):
    var b = u64(bits)
    var words = (size + 63) // 64
    if words >= PARALLEL_WORD_THRESHOLD:
        var chunk = words_chunk(words, PARALLEL_TASKS)
        def work(task: Int) {imm}:
            var offset = task * chunk
            var stop = min(offset + chunk, words)
            if offset < stop:
                invert_words(b.unsafe_offset(offset), 0, stop - offset)
        parallelize(work, PARALLEL_TASKS, PARALLEL_TASKS)
    else:
        invert_words(b, 0, words)
    # After the whole parallel region: no worker may touch this word unseen.
    if size % 64 != 0:
        b[words - 1] &= low_mask(size % 64)


@export("mbx_score_texts")
def mbx_score_texts(text1: Int, text2: Int, n: Int, table: Int, width: Int, gap1: Int, gap2: Int, gap_open: Float64, gap_extend: Float64) abi("C") -> Float64:
    var a = u8(text1)
    var b = u8(text2)
    var scores = fp(table)
    var total = 0.0
    var last_gap_a = False
    var last_gap_b = False
    for i in range(n):
        var ca = Int(a[i])
        var cb = Int(b[i])
        if ca == gap1 and cb == gap2:
            continue
        elif ca == gap1:
            total -= gap_extend
            if not last_gap_a:
                total -= gap_open
                last_gap_a = True
                last_gap_b = False
        elif cb == gap2:
            total -= gap_extend
            if not last_gap_b:
                total -= gap_open
                last_gap_a = False
                last_gap_b = True
        else:
            total += scores[ca * width + cb]
            last_gap_a = False
            last_gap_b = False
    return total


@export("mbx_accumulate_scores")
def mbx_accumulate_scores(text1: Int, text2: Int, n: Int, table: Int, width: Int, gap1: Int, gap2: Int, gap_open: Float64, gap_extend: Float64, skip_ref_gaps: Int, dst: Int) abi("C") -> Int:
    var a = u8(text1)
    var b = u8(text2)
    var scores = fp(table)
    var result = fp(dst)
    var total = 0.0
    var pos = 0
    var last_gap_a = False
    var last_gap_b = False
    for i in range(n):
        var ca = Int(a[i])
        var cb = Int(b[i])
        if ca == gap1 and cb == gap2:
            pass
        elif ca == gap1:
            total -= gap_extend
            if not last_gap_a:
                total -= gap_open
                last_gap_a = True
                last_gap_b = False
        elif cb == gap2:
            total -= gap_extend
            if not last_gap_b:
                total -= gap_open
                last_gap_a = False
                last_gap_b = True
        else:
            total += scores[ca * width + cb]
            last_gap_a = False
            last_gap_b = False
        if skip_ref_gaps == 0 or ca != gap1:
            result[pos] = total
            pos += 1
    return pos
