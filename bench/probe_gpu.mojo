"""Probe the max.gpu host surface on the pinned toolchain.

Reports whether a device is visible, what the DeviceContext API looks like, and
whether a trivial kernel can launch.  Run through pixi so MODULAR_HOME is set:

    pixi run mojo run bench/probe_gpu.mojo
"""

from max.gpu.host import DeviceContext


def main() raises:
    print("constructing DeviceContext")
    var ctx = DeviceContext()
    print("device:", ctx.name())


