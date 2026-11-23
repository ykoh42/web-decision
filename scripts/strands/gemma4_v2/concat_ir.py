"""Relax concatenation for split-decoder output assembly."""

import tvm
from tvm import relax


def build():
    n = tvm.tirx.Var("left_length", "int64")
    m = tvm.tirx.Var("right_length", "int64")
    left = relax.Var("left", relax.TensorType((1, n, 1536), "float32"))
    right = relax.Var("right", relax.TensorType((1, m, 1536), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [left, right]):
        with bb.dataflow():
            result = bb.emit(relax.op.concat((left, right), axis=1))
            result = bb.emit_output(result)
        bb.emit_func_output(result)
    return bb.get()
