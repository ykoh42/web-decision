"""Model-specific TVM graph construction."""

import tvm
from tvm import relax
from tvm.script import ir as I
from tvm.script import tirx as T


@I.ir_module(s_tir=True)
class Lookup:
    @T.prim_func(s_tir=True)
    def word(rows_h: T.handle, ids_h: T.handle, out_h: T.handle):
        n = T.int32()
        u = T.int32()
        rows = T.match_buffer(rows_h, (u, 768), "uint32")
        ids = T.match_buffer(ids_h, (n,), "int32")
        out = T.match_buffer(out_h, (1, n, 1536), "float32")
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
        for chunk in T.thread_binding(T.ceildiv(n, 32768), thread="blockIdx.y"):
            for pos in T.thread_binding(T.min(n, 32768), thread="blockIdx.x"):
                for lane in T.thread_binding(256, thread="threadIdx.x"):
                    row = chunk * 32768 + pos
                    if row < n:
                        for group in range(6):
                            column = group * 256 + lane
                            word = rows[ids[row], column // 2]
                            half = T.if_then_else(
                                column % 2 == 0,
                                word & T.uint32(65535),
                                word >> T.uint32(16),
                            )
                            value = T.reinterpret(
                                "float32", half << T.uint32(16)
                            ) * T.float32(39.25)
                            bits = T.reinterpret("uint32", value)
                            rounded = (
                                bits
                                + T.uint32(32767)
                                + ((bits >> T.uint32(16)) & T.uint32(1))
                            ) & T.uint32(4294901760)
                            out[0, row, column] = T.reinterpret("float32", rounded)


def build():
    n = tvm.tirx.Var("length", "int64")
    u = tvm.tirx.Var("unique", "int64")
    rows = relax.Var("rows", relax.TensorType((u, 768), "uint32"))
    ids = relax.Var("ids", relax.TensorType((n,), "int32"))
    builder = relax.BlockBuilder(Lookup)
    with builder.function(
        "packed_main",
        [rows, ids],
        attrs={
            "tir_var_lower_bound": {"length": 1, "unique": 1},
            "tir_var_upper_bound": {"length": 131072, "unique": 262144},
        },
    ):
        with builder.dataflow():
            output = builder.emit(
                relax.call_tir(
                    builder.get().get_global_var("word"),
                    (rows, ids),
                    relax.TensorType((1, n, 1536), "float32"),
                )
            )
            output = builder.emit_output(output)
        builder.emit_func_output(output)
    return builder.get()
