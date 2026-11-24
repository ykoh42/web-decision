"""Model-specific TVM graph construction."""

import tvm
from tvm import relax
from tvm.script import tirx as T


@T.macro
def rotary_row(ids, freq, out, row, lane, dim, trig):
    for group in range(dim // 256):
        col = group * 256 + lane
        angle = T.cast(ids[row], "float32") * freq[col % (dim // 2)]
        value = trig(angle)
        bits = T.reinterpret("uint32", value)
        rounded = (
            bits + T.uint32(32767) + ((bits >> T.uint32(16)) & T.uint32(1))
        ) & T.uint32(4294901760)
        out[0, row, col] = T.reinterpret("float32", rounded)


def kernel(dim, is_sine):
    trig = T.sin if is_sine else T.cos

    @T.prim_func(s_tir=True)
    def rotary(
        ids_h: T.handle, freq: T.Buffer((dim // 2,), "float32"), out_h: T.handle
    ):
        n = T.int32()
        ids = T.match_buffer(ids_h, (n,), "int32")
        out = T.match_buffer(out_h, (1, n, dim), "float32")
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
        for chunk in T.thread_binding(T.ceildiv(n, 32768), thread="blockIdx.y"):
            for pos in T.thread_binding(T.min(n, 32768), thread="blockIdx.x"):
                for lane in T.thread_binding(256, thread="threadIdx.x"):
                    row = chunk * 32768 + pos
                    if row < n:
                        rotary_row(ids, freq, out, row, lane, dim, trig)

    return rotary


def build():
    n = tvm.tirx.Var("length", "int64")
    bb = relax.BlockBuilder()
    for name, dim, sine in [
        ("local_cos", 256, False),
        ("local_sin", 256, True),
        ("global_cos", 512, False),
        ("global_sin", 512, True),
    ]:
        func = bb.add_func(
            kernel(dim, sine).with_attr("global_symbol", name + "_kernel"),
            name + "_kernel",
        )
        ids = relax.Var("ids", relax.TensorType((n,), "int32"))
        freq = relax.Var("freq", relax.TensorType((dim // 2,), "float32"))
        with bb.function(
            name,
            [ids, freq],
            attrs={
                "tir_var_lower_bound": {"length": 1},
            },
        ):
            with bb.dataflow():
                out = bb.emit(
                    relax.call_tir(
                        func, (ids, freq), relax.TensorType((1, n, dim), "float32")
                    )
                )
                out = bb.emit_output(out)
            bb.emit_func_output(out)
    return bb.get()
