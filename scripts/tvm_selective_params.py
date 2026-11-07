"""Keep tiny ONNX constants inside Relax; expose only large weights as inputs.

This is a process-local experiment for fixed shapes, not an upstream TVM patch.
"""
import math
import numpy as np
from tvm import relax
from tvm.relax.frontend.onnx import onnx_frontend

def parse_initializers(self, graph):
    for tensor in graph.initializer:
        if not tensor.name.strip(): raise ValueError('Tensor name is required')
        array=self._parse_array(tensor)
        nbytes=math.prod(array.shape)*np.dtype(str(array.dtype)).itemsize
        if self._keep_params_in_input and (nbytes>=1024*1024 or tensor.name.startswith(('q4_packed_', 'q4_scale_'))):
            name='weight_' + tensor.name.removeprefix('onnx::').lstrip('_')
            var=self._new_var(name,shape=array.shape,dtype=array.dtype)
            self._nodes[tensor.name]=var
            self._params[name]=(var,array)
        else:
            self._nodes[tensor.name]=relax.const(array)

onnx_frontend.ONNXGraphImporter._parse_graph_initializers=parse_initializers
