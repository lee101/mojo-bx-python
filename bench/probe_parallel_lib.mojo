"""Probe parallelize from a ctypes host: does it spawn, and is the work correct?"""

from max.algorithm import parallelize


def _fill(p: Int, n: Int):
    var ptr = Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=p)
    def work(i: Int) {imm}:
        ptr.unsafe_store(i, i * 2)
    parallelize(work, n, 4)


@export("probe_parallel_fill")
def probe_parallel_fill(p: Int, n: Int) abi("C") -> Int:
    _fill(p, n)
    var ptr = Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=p)
    var total = 0
    for i in range(n):
        total += Int(ptr.unsafe_load(i))
    return total


@export("probe_parallel_fill_2arg")
def probe_parallel_fill_2arg(p: Int, n: Int) abi("C") -> Int:
    var ptr = Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=p)
    def work(i: Int) {imm}:
        ptr.unsafe_store(i, i * 2)
    parallelize(work, n)
    var total = 0
    for i in range(n):
        total += Int(ptr.unsafe_load(i))
    return total
