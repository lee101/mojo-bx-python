"""C ABI kernels for the covered bx-python subset.

The bindings own every buffer.  Addresses are passed as Int because exported
Mojo functions cannot carry a pointer origin in their public signature.
"""

from std.sys import simd_width_of

comptime U8Ptr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int, AnyOrigin[mut=True]]
comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime BYTE_W = simd_width_of[DType.uint8]()


def u8(addr: Int) -> U8Ptr:
    return U8Ptr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


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


@export("mbx_bits_set_range")
def mbx_bits_set_range(bits: Int, start: Int, count: Int, value: Int) abi("C"):
    var b = u8(bits)
    var v = UInt8(0)
    if value != 0:
        v = UInt8(1)
    var vector_end = count - count % BYTE_W
    var values = SIMD[DType.uint8, BYTE_W](v)
    for i in range(0, vector_end, BYTE_W):
        b.store[width=BYTE_W, alignment=1](start + i, values)
    for i in range(vector_end, count):
        b[start + i] = v


@export("mbx_bits_count_range")
def mbx_bits_count_range(bits: Int, start: Int, count: Int) abi("C") -> Int:
    var b = u8(bits)
    var total = 0
    var vector_end = count - count % BYTE_W
    for i in range(0, vector_end, BYTE_W):
        var values = b.load[width=BYTE_W, alignment=1](start + i)
        total += Int(values.reduce_add())
    for i in range(vector_end, count):
        total += Int(b[start + i])
    return total


@export("mbx_bits_next")
def mbx_bits_next(bits: Int, start: Int, size: Int, value: Int) abi("C") -> Int:
    var b = u8(bits)
    var wanted = UInt8(0)
    if value != 0:
        wanted = UInt8(1)
    for i in range(start, size):
        if b[i] == wanted:
            return i
    return size


@export("mbx_bits_binary")
def mbx_bits_binary(bits: Int, other: Int, size: Int, operation: Int) abi("C"):
    var a = u8(bits)
    var b = u8(other)
    var vector_end = size - size % BYTE_W
    for i in range(0, vector_end, BYTE_W):
        var left = a.load[width=BYTE_W, alignment=1](i)
        var right = b.load[width=BYTE_W, alignment=1](i)
        if operation == 0:
            a.store[width=BYTE_W, alignment=1](i, left & right)
        elif operation == 1:
            a.store[width=BYTE_W, alignment=1](i, left | right)
        else:
            a.store[width=BYTE_W, alignment=1](i, left ^ right)
    for i in range(vector_end, size):
        if operation == 0:
            a[i] = a[i] & b[i]
        elif operation == 1:
            a[i] = a[i] | b[i]
        else:
            a[i] = a[i] ^ b[i]


@export("mbx_bits_invert")
def mbx_bits_invert(bits: Int, size: Int) abi("C"):
    var b = u8(bits)
    var vector_end = size - size % BYTE_W
    var ones = SIMD[DType.uint8, BYTE_W](UInt8(1))
    for i in range(0, vector_end, BYTE_W):
        b.store[width=BYTE_W, alignment=1](i, ones ^ b.load[width=BYTE_W, alignment=1](i))
    for i in range(vector_end, size):
        b[i] = UInt8(1) ^ b[i]


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
