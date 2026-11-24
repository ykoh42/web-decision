"""Insert one image's projected features into dynamic Gemma 4 text embeddings."""

import tvm
from tvm import relax
from tvm.script import ir as I
from tvm.script import tirx as T


@T.macro
def insert_value(
    hidden: T.Buffer,
    image: T.Buffer,
    start: T.Buffer,
    output: T.Buffer,
    row: T.int32,
    col: T.int32,
    k: T.int32,
):
    if row >= start[0] and row < start[0] + k:
        bits = T.reinterpret("uint32", image[row - start[0], col])
        if (bits & T.uint32(2139095040)) == T.uint32(2139095040) and (
            bits & T.uint32(8388607)
        ) != T.uint32(0):
            output[0, row, col] = T.reinterpret(
                "float32", (bits & T.uint32(2139095040)) | T.uint32(4194304)
            )
        else:
            rounded = (
                bits + T.uint32(32767) + ((bits >> T.uint32(16)) & T.uint32(1))
            ) & T.uint32(4294901760)
            output[0, row, col] = T.reinterpret("float32", rounded)
    else:
        output[0, row, col] = hidden[0, row, col]


@I.ir_module(s_tir=True)
class GpuKernels:
    @T.prim_func(s_tir=True)
    def insert(
        hidden_h: T.handle,
        image_h: T.handle,
        start: T.Buffer((1,), "int32"),
        output_h: T.handle,
    ):
        n = T.int32()
        k = T.int32()
        hidden = T.match_buffer(hidden_h, (1, n, 1536), "float32")
        image = T.match_buffer(image_h, (k, 1536), "float32")
        output = T.match_buffer(output_h, (1, n, 1536), "float32")
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
        for chunk in T.thread_binding(T.ceildiv(n, 32768), thread="blockIdx.y"):
            for pos in T.thread_binding(T.min(n, 32768), thread="blockIdx.x"):
                for lane in T.thread_binding(256, thread="threadIdx.x"):
                    row = chunk * 32768 + pos
                    if row < n:
                        for group in range(6):
                            col = group * 256 + lane
                            insert_value(hidden, image, start, output, row, col, k)


@I.ir_module(s_tir=True)
class CpuKernels:
    @T.prim_func(s_tir=True)
    def insert(
        hidden_h: T.handle,
        image_h: T.handle,
        start: T.Buffer((1,), "int32"),
        output_h: T.handle,
    ):
        n = T.int32()
        k = T.int32()
        hidden = T.match_buffer(hidden_h, (1, n, 1536), "float32")
        image = T.match_buffer(image_h, (k, 1536), "float32")
        output = T.match_buffer(output_h, (1, n, 1536), "float32")
        T.func_attr({"tirx.noalias": True})
        for row, col in T.grid(n, 1536):
            insert_value(hidden, image, start, output, row, col, k)


def build_image_embedding_ir(*, cpu=False):
    """Return a Relax module: (hidden[1,n,1536], image[k,1536], start[1]) → [1,n,1536].

    Both tensors are float32 and start is int32. The caller must validate
    1 <= k <= n and 0 <= start <= n-k. Call again on the result
    for each following image. `cpu=True` is only for reference comparisons.
    """
    n = tvm.tirx.Var("n", "int64")
    k = tvm.tirx.Var("k", "int64")
    builder = relax.BlockBuilder(CpuKernels if cpu else GpuKernels)
    hidden = relax.Var("hidden", relax.TensorType((1, n, 1536), "float32"))
    image = relax.Var("image", relax.TensorType((k, 1536), "float32"))
    start = relax.Var("start", relax.TensorType((1,), "int32"))
    attributes = {
        "tir_var_lower_bound": {"n": 1, "k": 1},
    }
    with builder.function("main", [hidden, image, start], attrs=attributes):
        with builder.dataflow():
            result = builder.emit(
                relax.call_tir(
                    builder.get().get_global_var("insert"),
                    (hidden, image, start),
                    relax.TensorType((1, n, 1536), "float32"),
                )
            )
            result = builder.emit_output(result)
        builder.emit_func_output(result)
    return builder.get()
