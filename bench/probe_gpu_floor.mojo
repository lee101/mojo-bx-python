"""Fixed GPU cost per call: allocate, copy up, copy down, synchronise, no kernel.

Sizes match the heaviest caller case for score_texts (100 M columns x 2 uint8
streams = 100 MB up) and for the bitset reads (1 GiB = 1.07 GB up and back).
No kernel runs, so every number here is a floor that a kernel can only add to.
"""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext


def now() -> Float64:
    return Float64(perf_counter_ns()) / 1e9


def bench_copy(ctx: DeviceContext, nbytes: Int, reps: Int) raises:
    var host_in = List[UInt8]()
    host_in.resize(unsafe_uninit_length=nbytes)
    var dev = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    var total = 0.0
    for _ in range(reps):
        var s = now()
        ctx.enqueue_copy(dev, host_in)
        ctx.synchronize()
        total += now() - s
    print("upload", Float64(nbytes) / 1e6, "MB:", total / Float64(reps) * 1000.0, "ms")
    var host_out = List[UInt8]()
    host_out.resize(unsafe_uninit_length=nbytes)
    var total2 = 0.0
    for _ in range(reps):
        var s = now()
        ctx.enqueue_copy(host_out, dev)
        ctx.synchronize()
        total2 += now() - s
    print("download", Float64(nbytes) / 1e6, "MB:", total2 / Float64(reps) * 1000.0, "ms")


def main() raises:
    var ctx = DeviceContext()
    print("device:", ctx.name())
    # warm up the context/allocator so the first call is not in the average
    bench_copy(ctx, 1_000_000, 1)
    bench_copy(ctx, 200_000_000, 5)   # 100M cols x 2 uint8 streams
    bench_copy(ctx, 1_073_741_824, 3) # 1 GiB stream
