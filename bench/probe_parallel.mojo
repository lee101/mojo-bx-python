from std.sys import num_physical_cores
from max.algorithm import parallelize


def work(i: Int):
    pass


def main() raises:
    print("physical cores:", num_physical_cores())
    try:
        parallelize(work, 64)
        print("parallelize(func, n): ok")
    except e:
        print("parallelize(func, n) FAILED:", e)
    try:
        parallelize(work, 64, 4)
        print("parallelize(func, n, 4): ok")
    except e:
        print("parallelize(func, n, 4) FAILED:", e)
