"""Narrow bounded model integer tensors to int32 for this WebGPU experiment.

The selected Kev fixture uses token IDs below 152k and only 69 positions. This
is not a general ONNX conversion and requires checking all supported input bounds.
"""
import numpy as np
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import onnx_frontend

original=onnx_frontend.ONNXGraphImporter._parse_graph_initializers

def narrowed(self,graph):
    original_parse_array=self._parse_array
    def parse(tensor):
        out=original_parse_array(tensor)
        if str(out.dtype)=='bool':
            return tvm.runtime.tensor(out.numpy().astype(np.int32))
        if str(out.dtype)=='int64':
            data=out.numpy()
            if data.size==0 or (data.min()>=np.iinfo(np.int32).min and data.max()<=np.iinfo(np.int32).max):
                return tvm.runtime.tensor(data.astype(np.int32))
        return out
    self._parse_array=parse
    try:original(self,graph)
    finally:self._parse_array=original_parse_array

onnx_frontend.ONNXGraphImporter._parse_graph_initializers=narrowed
original_cast=onnx_frontend.Cast._impl_v13.__func__

def cast(cls,bb,inputs,attrs,params):
    if attrs.get('to')==7:
        x=inputs[0]
        if isinstance(x,tvm.ir.GenericConst):return relax.const(x.value.numpy().astype(np.int32))
        return relax.op.astype(x,'int32')
    return original_cast(cls,bb,inputs,attrs,params)

onnx_frontend.Cast._impl_v13=classmethod(cast)


# WGSL cannot store bool/i8 arrays. Keep boolean intermediates as int32 and
# convert to predicates only inside a fused where/comparison operation.
original_equal=onnx_frontend.Equal._impl_v13.__func__
def equal(cls,bb,inputs,attrs,params):
    result=original_equal(cls,bb,inputs,attrs,params)
    if isinstance(result,tvm.ir.GenericConst):return relax.const(result.value.numpy().astype(np.int32))
    return relax.op.astype(result,'int32')
onnx_frontend.Equal._impl_v13=classmethod(equal)

def and_op(cls,bb,inputs,attrs,params):return relax.op.multiply(inputs[0],inputs[1])
def or_op(cls,bb,inputs,attrs,params):return relax.op.maximum(inputs[0],inputs[1])
def not_op(cls,bb,inputs,attrs,params):return relax.op.subtract(relax.const(1,'int32'),inputs[0])
onnx_frontend.And._impl_v1=classmethod(and_op)
onnx_frontend.Or._impl_v1=classmethod(or_op)
onnx_frontend.Not._impl_v1=classmethod(not_op)
original_where=onnx_frontend.Where._impl_v16.__func__
def where(cls,bb,inputs,attrs,params):
    inputs=list(inputs)
    if str(inputs[0].ty.dtype)=='int32':
        inputs[0]=relax.op.not_equal(inputs[0],relax.const(0,'int32'))
    return original_where(cls,bb,inputs,attrs,params)
onnx_frontend.Where._impl_v16=classmethod(where)
