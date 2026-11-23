"""Scratch-only split of the model-local slice into two single-output calls."""

import tvm
from tvm import relax
from tvm.script import tirx as T


@T.prim_func(s_tir=True)
def hidden_kernel(hidden_h: T.handle, start: T.Buffer((1,), "int32"), out_h: T.handle):
    n = T.int32()
    chunk = T.int32()
    hidden = T.match_buffer(hidden_h, (1, n, 1536), "float32")
    out = T.match_buffer(out_h, (1, chunk, 1536), "float32")
    T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
    for row in T.thread_binding(chunk, thread="blockIdx.x"):
        for lane in T.thread_binding(256, thread="threadIdx.x"):
            for group in range(6):
                out[0, row, group * 256 + lane] = hidden[
                    0, start[0] + row, group * 256 + lane
                ]


@T.prim_func(s_tir=True)
def per_kernel(per_h: T.handle, start: T.Buffer((1,), "int32"), out_h: T.handle):
    n = T.int32()
    chunk = T.int32()
    per = T.match_buffer(per_h, (1, n, 256), "float32")
    out = T.match_buffer(out_h, (1, chunk, 256), "float32")
    T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
    for row in T.thread_binding(chunk, thread="blockIdx.x"):
        for lane in T.thread_binding(256, thread="threadIdx.x"):
            out[0, row, lane] = per[0, start[0] + row, lane]


def build():
    bb = relax.BlockBuilder()
    k_hidden = bb.add_func(hidden_kernel, "hidden_kernel")
    k_per = bb.add_func(per_kernel, "per_kernel")

    def add_slice(name, width, kernel):
        n = tvm.tirx.Var("n_" + name, "int64")
        c = tvm.tirx.Var("c_" + name, "int64")
        x = relax.Var("input", relax.TensorType((1, n, width), "float32"))
        start = relax.Var("start", relax.TensorType((1,), "int32"))
        template = relax.Var("template", relax.TensorType((1, c, 1), "float32"))
        with bb.function(name, [x, start, template]):
            with bb.dataflow():
                out = bb.emit_output(
                    relax.call_tir(
                        kernel, (x, start), relax.TensorType((1, c, width), "float32")
                    )
                )
            bb.emit_func_output(out)
        return bb.get().get_global_var(name)

    g_hidden = add_slice("hidden", 1536, k_hidden)
    g_per = add_slice("per", 256, k_per)
    n = tvm.tirx.Var("n", "int64")
    c = tvm.tirx.Var("c", "int64")
    h = relax.Var("hidden", relax.TensorType((1, n, 1536), "float32"))
    p = relax.Var("per_layer", relax.TensorType((1, n, 256), "float32"))
    start = relax.Var("start", relax.TensorType((1,), "int32"))
    template = relax.Var("template", relax.TensorType((1, c, 1), "float32"))
    with bb.function("main", [h, p, start, template]):
        with bb.dataflow():
            h_out = bb.emit(relax.Call(g_hidden, [h, start, template]))
            p_out = bb.emit(relax.Call(g_per, [p, start, template]))
            result = bb.emit_output(relax.Tuple([h_out, p_out]))
        bb.emit_func_output(result)
    return bb.get()
