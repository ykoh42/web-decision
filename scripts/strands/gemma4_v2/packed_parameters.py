"""Pack BF16 checkpoint parameters for model-specific TVM graphs."""

import numpy as np
import tvm
from tvm import relax
from tvm.script import tirx as T


def make_unpack_matrix(shape):
    size = int(np.prod(shape))
    words = (size + 1) // 2
    rows, columns = shape

    @T.prim_func(s_tir=True)
    def unpack_matrix(
        weight: T.Buffer((words,), "uint32"),
        output: T.Buffer((rows, columns), "float32"),
    ):
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
        with T.sblock("root"):
            for block in T.thread_binding((size + 255) // 256, thread="blockIdx.x"):
                for thread in T.thread_binding(256, thread="threadIdx.x"):
                    index = block * 256 + thread
                    if index < size:
                        word = weight[index // 2]
                        half = T.if_then_else(
                            index % 2 == 0,
                            word & T.uint32(65535),
                            word >> T.uint32(16),
                        )
                        output[index // columns, index % columns] = T.reinterpret(
                            "float32", half << T.uint32(16)
                        )

    return unpack_matrix


def make_unpack_vector(shape):
    size = int(np.prod(shape))
    words = (size + 1) // 2

    @T.prim_func(s_tir=True)
    def unpack(
        weight: T.Buffer((words,), "uint32"), output: T.Buffer((size,), "float32")
    ):
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
        with T.sblock("root"):
            for block in T.thread_binding((size + 255) // 256, thread="blockIdx.x"):
                for thread in T.thread_binding(256, thread="threadIdx.x"):
                    index = block * 256 + thread
                    if index < size:
                        word = weight[index // 2]
                        half = T.if_then_else(
                            index % 2 == 0, word & T.uint32(65535), word >> T.uint32(16)
                        )
                        output[index] = T.reinterpret("float32", half << T.uint32(16))

    return unpack


def make_unpack(shape):
    if len(shape) == 2:
        return make_unpack_matrix(shape)
    if len(shape) == 1:
        return make_unpack_vector(shape)
    raise ValueError(f"Unsupported parameter shape {shape}")


def pack_parameters(module, parameters, input_count):
    original = module["main"]
    print(
        "Packing graph arguments",
        len(original.params),
        input_count,
        len(parameters),
        flush=True,
    )
    module = tvm.IRModule(
        {
            key: value
            for key, value in module.functions.items()
            if key.name_hint != "main"
        },
        attrs=module.attrs,
        global_infos=module.global_infos,
    )
    builder = relax.BlockBuilder(module)
    float_main = builder.add_func(
        original.with_attr("global_symbol", "float_main"), "float_main"
    )
    kernels = {}
    tensors = []
    values = []
    for index, parameter in enumerate(parameters):
        array = parameter.numpy()
        bits = array.view(np.uint32).reshape(-1)
        if np.any(bits & np.uint32(65535)):
            raise ValueError(f"Parameter {index} is not exactly BF16-representable")
        halves = (bits >> np.uint32(16)).astype(np.uint16)
        if len(halves) % 2:
            halves = np.pad(halves, (0, 1))
        values.append(halves.view(np.uint32))
        shape = array.shape
        name = "unpack_" + "_".join(str(dimension) for dimension in shape)
        if shape not in kernels:
            kernels[shape] = builder.add_func(
                make_unpack(shape).with_attr("global_symbol", name),
                name,
            )
        tensors.append(shape)
    variables = [
        relax.Var(f"input_{index}", parameter.ty)
        for index, parameter in enumerate(original.params[:input_count])
    ] + [
        relax.Var(f"packed_{index}", relax.TensorType(value.shape, "uint32"))
        for index, value in enumerate(values)
    ]
    with builder.function("packed_main", variables):
        with builder.dataflow():
            decoded = []
            for variable, shape in zip(variables[input_count:], tensors):
                decoded.append(
                    builder.emit(
                        relax.call_tir(
                            kernels[shape],
                            (variable,),
                            relax.TensorType(shape, "float32"),
                        )
                    )
                )
            result = builder.emit(
                relax.Call(float_main, variables[:input_count] + decoded)
            )
            result = builder.emit_output(result)
        builder.emit_func_output(result)
    return builder.get(), values
