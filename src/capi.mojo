"""C ABI kernels for the covered bx-python subset.

The bindings own every buffer.  Addresses are passed as Int because exported
Mojo functions cannot carry a pointer origin in their public signature.
"""

from max.algorithm import parallelize
from std.bit import count_trailing_zeros, pop_count
from std.sys import simd_width_of as simdwidthof

comptime U8Ptr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime U64Ptr = UnsafePointer[UInt64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int, AnyOrigin[mut=True]]
comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime W = simdwidthof[DType.float64]()
comptime PARALLEL_WORD_THRESHOLD = 1 << 20
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
    """Filter a previously bounded interval window and write relative indices."""
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
    var e = ip(ends)
    var result = ip(dst)
    var count = 0
    for i in range(lo, hi):
        if e[i] > start:
            result[count] = i
            count += 1
    return count


@export("mbx_overlap_counts")
def mbx_overlap_counts(starts: Int, ends: Int, n: Int, qstarts: Int, qends: Int, m: Int, dst: Int) abi("C"):
    """Count overlaps for many queries against start-sorted intervals."""
    var s = ip(starts)
    var e = ip(ends)
    var qs = ip(qstarts)
    var qe = ip(qends)
    var result = ip(dst)
    for q in range(m):
        var count = 0
        for i in range(n):
            if s[i] >= qe[q]:
                break
            if e[i] > qs[q]:
                count += 1
        result[q] = count


def low_mask(bits: Int) -> UInt64:
    if bits >= 64:
        return ~UInt64(0)
    return (UInt64(1) << UInt64(bits)) - UInt64(1)


def fill_words(b: U64Ptr, start: Int, count: Int, value: UInt64):
    var vector_end = count - count % W
    var values = SIMD[DType.uint64, W](value)
    for i in range(0, vector_end, W):
        b.store(start + i, values)
    for i in range(vector_end, count):
        b[start + i] = value


def fill_words_parallel(b: U64Ptr, start: Int, count: Int, value: UInt64):
    var chunk = (count + PARALLEL_TASKS - 1) // PARALLEL_TASKS
    chunk = (chunk + W - 1) // W * W

    @__parameter
    def work(task: Int):
        var offset = task * chunk
        var stop = min(offset + chunk, count)
        if offset < stop:
            fill_words(b, start + offset, stop - offset, value)

    parallelize[work](PARALLEL_TASKS, PARALLEL_TASKS)


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
    if middle_count >= PARALLEL_WORD_THRESHOLD:
        fill_words_parallel(b, middle_start, middle_count, fill)
    else:
        fill_words(b, middle_start, middle_count, fill)


@export("mbx_bits_count_range")
def mbx_bits_count_range(bits: Int, start: Int, count: Int) abi("C") -> Int:
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
    var vector_end = middle_count - middle_count % W
    for i in range(0, vector_end, W):
        total += Int(pop_count(b.load[width=W](middle_start + i)).reduce_add())
    for i in range(vector_end, middle_count):
        total += Int(pop_count(b[middle_start + i]))
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
        var chunk = (words + PARALLEL_TASKS - 1) // PARALLEL_TASKS
        chunk = (chunk + W - 1) // W * W

        @__parameter
        def work(task: Int):
            var start = task * chunk
            var stop = min(start + chunk, words)
            if start < stop:
                bits_binary_words(a, b, start, stop - start, operation)

        parallelize[work](PARALLEL_TASKS, PARALLEL_TASKS)
    else:
        bits_binary_words(a, b, 0, words, operation)


def bits_binary_words(a: U64Ptr, b: U64Ptr, start: Int, count: Int, operation: Int):
    var vector_end = count - count % W
    for i in range(0, vector_end, W):
        var offset = start + i
        var left = a.load[width=W](offset)
        var right = b.load[width=W](offset)
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
        b.store(offset, ~b.load[width=W](offset))
    for i in range(vector_end, count):
        b[start + i] = ~b[start + i]


@export("mbx_bits_invert")
def mbx_bits_invert(bits: Int, size: Int) abi("C"):
    var b = u64(bits)
    var words = (size + 63) // 64
    if words >= PARALLEL_WORD_THRESHOLD:
        var chunk = (words + PARALLEL_TASKS - 1) // PARALLEL_TASKS
        chunk = (chunk + W - 1) // W * W

        @__parameter
        def work(task: Int):
            var start = task * chunk
            var stop = min(start + chunk, words)
            if start < stop:
                invert_words(b, start, stop - start)

        parallelize[work](PARALLEL_TASKS, PARALLEL_TASKS)
    else:
        invert_words(b, 0, words)
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
