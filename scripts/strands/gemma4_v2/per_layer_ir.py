"""Build the packed per-layer embedding projection graph."""

import torch
import transformers
from packed_parameters import pack_parameters
from safetensors import safe_open
from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
from tvm.script import ir as I
from tvm.script import tirx as T


@I.ir_module(s_tir=True)
class Gather:
    @T.prim_func(s_tir=True)
    def gather(ids_h: T.handle, rows_h: T.handle, output_h: T.handle):
        n = T.int32()
        u = T.int32()
        ids = T.match_buffer(ids_h, (n,), "int32")
        rows = T.match_buffer(rows_h, (u, 128), "uint32")
        out = T.match_buffer(output_h, (1, n, 256), "float32")
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
        for chunk in T.thread_binding(T.ceildiv(n, 32768), thread="blockIdx.y"):
            for block in T.thread_binding(T.min(n, 32768), thread="blockIdx.x"):
                for column in T.thread_binding(256, thread="threadIdx.x"):
                    row = chunk * 32768 + block
                    if row < n:
                        word = rows[ids[row], column // 2]
                        half = T.if_then_else(
                            column % 2 == 0,
                            word & T.uint32(65535),
                            word >> T.uint32(16),
                        )
                        value = T.reinterpret(
                            "float32", half << T.uint32(16)
                        ) * T.float32(16)
                        bits = T.reinterpret("uint32", value)
                        rounded = (
                            bits
                            + T.uint32(32767)
                            + ((bits >> T.uint32(16)) & T.uint32(1))
                        ) & T.uint32(4294901760)
                        out[0, row, column] = T.reinterpret("float32", rounded)


@torch.library.custom_op("webdecision::ple_gather", mutates_args=())
def gather_ids(indices: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    words = rows.to(torch.int64)[indices.to(torch.int64)]
    bits = (
        torch.stack((words & 65535, words >> 16), dim=-1)
        .reshape(-1, 256)
        .to(torch.int32)
        << 16
    )
    return (bits.contiguous().view(torch.float32) * 16).reshape(1, -1, 256)


@gather_ids.register_fake
def gather_ids_fake(indices, rows):
    return torch.empty(
        (1, indices.shape[0], 256), device=rows.device, dtype=torch.float32
    )


class Projection(torch.nn.Module):
    def __init__(self, norm_eps, weights, norm):
        super().__init__()
        self.projection = torch.nn.Linear(1536, 256, bias=False)
        self.norm = Gemma4RMSNorm(256, eps=norm_eps)
        self.projection.weight.data.copy_(weights[:256])
        self.norm.weight.data.copy_(norm)

    def forward(self, hidden_states, indices, rows):
        return (
            self.norm(self.projection(hidden_states) * (1536**-0.5))
            + gather_ids(indices, rows)
        ) * (2**-0.5)


def convert_gather(node, importer):
    index, rows = (importer.env[arg] for arg in node.args)
    seq = index.ty.shape.values[0]
    kernel = importer.block_builder.add_func(Gather["gather"], "gather_ple")
    return importer.block_builder.emit(
        relax.call_tir(
            kernel, (index, rows), relax.TensorType((1, seq, 256), "float32")
        )
    )


def build_per_layer_ir(checkpoint):
    config = transformers.AutoConfig.from_pretrained(str(checkpoint)).text_config
    with safe_open(checkpoint / "model.safetensors", framework="pt") as source:
        weights = source.get_tensor(
            "model.language_model.per_layer_model_projection.weight"
        ).float()
        norm = source.get_tensor(
            "model.language_model.per_layer_projection_norm.weight"
        ).float()
    model = Projection(config.rms_norm_eps, weights, norm).eval()
    length = torch.export.Dim("length", min=2)
    count = torch.export.Dim("unique", min=2, max=262144)
    inputs = (
        torch.zeros((1, 80, 1536)),
        torch.arange(80, dtype=torch.int32),
        torch.zeros((80, 128), dtype=torch.uint32),
    )
    program = torch.export.export(
        model,
        inputs,
        dynamic_shapes=({1: length}, {0: length}, {0: count}),
        strict=False,
    )
    module = from_exported_program(
        program,
        unwrap_unit_return_tuple=True,
        keep_params_as_input=True,
        custom_convert_map={"ple_gather.default": convert_gather},
    )
    module, params = relax.frontend.detach_params(module)
    module, _ = pack_parameters(module, params["main"], 3)
    return module
