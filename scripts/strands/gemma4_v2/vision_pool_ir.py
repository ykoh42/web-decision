"""Model-specific TVM graph construction."""

from tvm.script import ir as I
from tvm.script import tirx as T


@I.ir_module(s_tir=True)
class Pool:
    @T.prim_func(s_tir=True)
    def pool(
        hidden: T.Buffer((1, 2520, 768), "float32"),
        indices_h: T.handle,
        output_h: T.handle,
    ):
        n = T.int32()
        indices = T.match_buffer(indices_h, (n, 9), "int32")
        output = T.match_buffer(output_h, (n, 768), "float32")
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
        for block in T.thread_binding(n * 3, thread="blockIdx.x"):
            for thread in T.thread_binding(256, thread="threadIdx.x"):
                index = block * 256 + thread
                row, column = index // 768, index % 768
                total = T.alloc_buffer((1,), "float32", scope="local")
                total[0] = T.float32(0)
                for item in range(9):
                    total[0] = total[0] + hidden[
                        0, indices[row, item], column
                    ] * T.float32(1 / 9)
                output[row, column] = total[0] * T.float32(768**0.5)
