"""Process-local ONNX shape handling for the bounded dynamic Kev experiment.

This is not a general-purpose ONNX frontend: only the graph's known shape
construction patterns and the int32 narrowing used by this experiment are
covered. Keep the original converter for all other cases.
"""
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import onnx_frontend
from tvm_narrow_int64 import original_where

original_concat = onnx_frontend.Concat._impl_v13.__func__
original_gather = onnx_frontend.Gather._impl_v13.__func__
original_range = onnx_frontend.Range._impl_v12.__func__
narrow_where = onnx_frontend.Where._impl_v16.__func__


def concat(cls, bb, inputs, attr, params):
    if attr.get('axis', 0) == 0 and any(isinstance(x, relax.ShapeExpr) for x in inputs):
        values = []
        for item in inputs:
            if isinstance(item, relax.ShapeExpr):
                values.extend(item.values)
            elif isinstance(item, tvm.ir.GenericConst) and item.value.numpy().ndim == 1 and str(item.ty.dtype) in ('int32', 'int64'):
                values.extend(int(v) for v in item.value.numpy())
            else:
                break
        else:
            return relax.ShapeExpr(values)
    return original_concat(cls, bb, inputs, attr, params)


onnx_frontend.Concat._impl_v13 = classmethod(concat)


def gather(cls, bb, inputs, attr, params):
    inputs = list(inputs)
    data = inputs[0]
    if not isinstance(data, relax.ShapeExpr) and isinstance(data.ty, relax.ShapeType):
        inputs[0] = bb.normalize(relax.op.shape_to_tensor(data))
    elif isinstance(inputs[1], tvm.ir.GenericConst) and not isinstance(data, (tvm.ir.GenericConst, relax.ShapeExpr)):
        indices = inputs[1].value.numpy()
        axis = attr.get('axis', 0)
        shape = data.ty.shape
        if shape is not None and indices.size and (indices >= 0).all():
            extent = shape[axis]
            # In this bounded experiment all dynamic sequence lengths are >= 1.
            # Only bypass index normalization when the index is provably valid.
            if (isinstance(extent, tvm.tirx.IntImm) and int(indices.max()) < int(extent)) or (not isinstance(extent, tvm.tirx.IntImm) and int(indices.max()) == 0):
                return relax.op.take(data, inputs[1], axis=axis, mode='fast')
    return original_gather(cls, bb, inputs, attr, params)


onnx_frontend.Gather._impl_v13 = classmethod(gather)


def range_op(cls, bb, inputs, attr, params):
    start, limit, step = inputs
    if (isinstance(start, tvm.ir.GenericConst) and int(start.value.numpy()) == 0
            and isinstance(step, tvm.ir.GenericConst) and int(step.value.numpy()) == 1
            and tvm.ir.is_prim_expr(limit)):
        # The length comes directly from the input shape.  Preserve that
        # relation instead of computing a new data-dependent shape tensor.
        return relax.op.arange(0, limit, 1, 'int32')
    return original_range(cls, bb, inputs, attr, params)


onnx_frontend.Range._impl_v12 = classmethod(range_op)


def where(cls, bb, inputs, attr, params):
    if any(isinstance(x, relax.ShapeExpr) for x in inputs) and all(isinstance(x, (relax.ShapeExpr, tvm.ir.GenericConst)) for x in inputs):
        return original_where(cls, bb, inputs, attr, params)
    return narrow_where(cls, bb, inputs, attr, params)


onnx_frontend.Where._impl_v16 = classmethod(where)


def _mask_int32(value):
    return relax.op.astype(value, 'int32') if str(value.ty.dtype) == 'bool' else value


def and_op(cls, bb, inputs, attr, params):
    return relax.op.multiply(_mask_int32(inputs[0]), _mask_int32(inputs[1]))


def or_op(cls, bb, inputs, attr, params):
    return relax.op.maximum(_mask_int32(inputs[0]), _mask_int32(inputs[1]))


def not_op(cls, bb, inputs, attr, params):
    return relax.op.subtract(relax.const(1, 'int32'), _mask_int32(inputs[0]))


onnx_frontend.And._impl_v1 = classmethod(and_op)
onnx_frontend.Or._impl_v1 = classmethod(or_op)
onnx_frontend.Not._impl_v1 = classmethod(not_op)
