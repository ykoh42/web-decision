"""Slice precomputed absolute RoPE rows for a query chunk."""

import tvm
from tvm import relax
from tvm.script import tirx as T


def build(width):
    @T.prim_func(s_tir=True)
    def slice_rope(
        source_h: T.handle, offset: T.Buffer((1,), "int32"), result_h: T.handle
    ):
        n = T.int32()
        q = T.int32()
        source = T.match_buffer(source_h, (1, n, width), "float32")
        result = T.match_buffer(result_h, (1, q, width), "float32")
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
        for row in T.thread_binding(q, thread="blockIdx.x"):
            for lane in T.thread_binding(256, thread="threadIdx.x"):
                for group in range(width // 256):
                    column = group * 256 + lane
                    result[0, row, column] = source[0, offset[0] + row, column]

    builder = relax.BlockBuilder()
    function = builder.add_func(slice_rope, "slice_rope")
    n, q = (tvm.tirx.Var(name, "int64") for name in ("n", "q"))
    cosine = relax.Var("cosine", relax.TensorType((1, n, width), "float32"))
    sine = relax.Var("sine", relax.TensorType((1, n, width), "float32"))
    offset = relax.Var("offset", relax.TensorType((1,), "int32"))
    template = relax.Var("template", relax.TensorType((1, q, 1), "float32"))
    with builder.function("main", [cosine, sine, offset, template]):
        with builder.dataflow():
            rows = [
                builder.emit(
                    relax.call_tir(
                        function,
                        (tensor, offset),
                        relax.TensorType((1, q, width), "float32"),
                    )
                )
                for tensor in (cosine, sine)
            ]
            output = builder.emit_output(relax.Tuple(rows))
        builder.emit_func_output(output)
    return builder.get()
