"""FP32 FFN with three pretransposed matrices reused across chunks."""

import math

import torch
from packed_parameters import make_unpack
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program


class TailPretransposed(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.pre = block.pre_feedforward_layernorm
        self.post = block.post_feedforward_layernorm
        self.gate = block.per_layer_input_gate
        self.projection = block.per_layer_projection
        self.per_norm = block.post_per_layer_input_norm
        self.mlp_activation = block.mlp.act_fn
        self.per_activation = block.act_fn

    def forward(self, hidden, per_layer, scale, gate_t, up_t, down_t):
        residual = hidden
        norm = self.pre(hidden)
        hidden = torch.matmul(
            self.mlp_activation(torch.matmul(norm, gate_t)) * torch.matmul(norm, up_t),
            down_t,
        )
        hidden = self.post(hidden) + residual
        residual = hidden
        hidden = self.per_activation(self.gate(hidden)) * per_layer
        hidden = self.per_norm(self.projection(hidden)) + residual
        return hidden * scale


def tail(block, hidden, per_layer, scale):
    width = block.mlp.gate_proj.weight.shape[0]
    weights = (
        torch.empty(1536, width),
        torch.empty(1536, width),
        torch.empty(width, 1536),
    )
    length = torch.export.Dim("chunk", min=1)
    program = torch.export.export(
        TailPretransposed(block).eval(),
        (hidden, per_layer, scale, *weights),
        dynamic_shapes=({1: length}, {1: length}, None, None, None, None),
        strict=False,
    )
    print(
        "optimized tail parameters",
        [name for name, _ in program.named_parameters()],
        flush=True,
    )
    mod = from_exported_program(
        program, unwrap_unit_return_tuple=True, keep_params_as_input=True
    )
    mod, params = relax.frontend.detach_params(mod)
    return mod, params["main"]


def unpack_transpose(parameters):
    bb = relax.BlockBuilder()
    inputs = []
    outputs = []
    for i, parameter in enumerate(parameters):
        shape = tuple(int(n) for n in parameter.shape)
        inputs.append(
            relax.Var(
                f"packed_{i}",
                relax.TensorType(((math.prod(shape) + 1) // 2,), "uint32"),
            )
        )
        outputs.append((shape, bb.add_func(make_unpack(shape), f"unpack_{i}")))
    with bb.function("main", inputs):
        with bb.dataflow():
            tensors = []
            for i, (value, (shape, fn)) in enumerate(zip(inputs, outputs)):
                tensor = bb.emit(
                    relax.call_tir(fn, (value,), relax.TensorType(shape, "float32"))
                )
                if i in (1, 2, 3):
                    tensor = bb.emit(relax.op.permute_dims(tensor, [1, 0]))
                tensors.append(tensor)
            result = bb.emit_output(relax.Tuple(tensors))
        bb.emit_func_output(result)
    return bb.get()
