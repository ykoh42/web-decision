"""Model-specific TVM graph construction."""

import torch
import transformers
from packed_parameters import pack_parameters
from safetensors import safe_open
from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
from tvm.script import tirx as T


@T.prim_func(s_tir=True)
def round_bf16(x: T.Buffer((1, 1536), "float32"), out: T.Buffer((1, 1536), "float32")):
    T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
    with T.sblock("root"):
        for block in T.thread_binding(6, thread="blockIdx.x"):
            for lane in T.thread_binding(256, thread="threadIdx.x"):
                i = block * 256 + lane
                bits = T.reinterpret("uint32", x[0, i])
                rounded = (
                    bits + T.uint32(32767) + ((bits >> T.uint32(16)) & T.uint32(1))
                ) & T.uint32(4294901760)
                is_nan = ((bits & T.uint32(0x7F800000)) == T.uint32(0x7F800000)) & (
                    (bits & T.uint32(0x007FFFFF)) != T.uint32(0)
                )
                out[0, i] = T.reinterpret(
                    "float32", T.if_then_else(is_nan, bits, rounded)
                )


@T.prim_func(s_tir=True)
def last_token(x_h: T.handle, out: T.Buffer((1, 1536), "float32")):
    n = T.int32()
    x = T.match_buffer(x_h, (1, n, 1536), "float32")
    T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
    with T.sblock("root"):
        for block in T.thread_binding(6, thread="blockIdx.x"):
            for lane in T.thread_binding(256, thread="threadIdx.x"):
                i = block * 256 + lane
                out[0, i] = x[0, n - 1, i]


@T.prim_func(s_tir=True)
def accumulate_readout(
    hidden: T.Buffer((1, 1536), "float32"),
    weights: T.Buffer((195840,), "uint32"),
    logits: T.Buffer((1, 255), "float32"),
):
    T.func_attr(
        {"tirx.is_host_func": True, "tirx.is_scheduled": True, "tirx.noalias": True}
    )
    for option in range(255):
        acc = T.alloc_buffer((1,), "float64", scope="local")
        acc[0] = T.float64(0)
        for feature in range(1536):
            word = weights[option * 768 + feature // 2]
            half = (word >> T.Cast("uint32", (feature % 2) * 16)) & T.uint32(65535)
            weight = T.reinterpret("float32", half << T.uint32(16))
            acc[0] += T.Cast("float64", hidden[0, feature]) * T.Cast("float64", weight)
        # Strands rounds the FP64 dot product to FP32 and then uses BF16 ties to even.
        bits = T.reinterpret("uint32", T.Cast("float32", acc[0]))
        rounded = (
            bits + T.uint32(32767) + ((bits >> T.uint32(16)) & T.uint32(1))
        ) & T.uint32(4294901760)
        is_nan = ((bits & T.uint32(0x7F800000)) == T.uint32(0x7F800000)) & (
            (bits & T.uint32(0x007FFFFF)) != T.uint32(0)
        )
        logits[0, option] = T.reinterpret(
            "float32", T.if_then_else(is_nan, bits, rounded)
        )


@torch.library.custom_op("webdecision::last_token", mutates_args=())
def last_torch(x: torch.Tensor) -> torch.Tensor:
    return x[:, -1, :].clone()


@last_torch.register_fake
def _(x):
    return torch.empty((1, 1536), device=x.device, dtype=x.dtype)


@torch.library.custom_op("webdecision::bf16_round", mutates_args=())
def round_torch(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


@round_torch.register_fake
def _(x):
    return torch.empty_like(x)


def build(checkpoint, hidden, *, host_readout=False):
    config = transformers.AutoConfig.from_pretrained(str(checkpoint)).text_config

    class Readout(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = Gemma4RMSNorm(1536, eps=config.rms_norm_eps)
            with safe_open(checkpoint / "model.safetensors", framework="pt") as f:
                self.norm.weight.data.copy_(
                    f.get_tensor("model.language_model.norm.weight").float()
                )
            if not host_readout:
                self.linear = torch.nn.Linear(1536, 255, bias=False)
                with safe_open(checkpoint / "readout.safetensors", framework="pt") as f:
                    self.linear.weight.data.copy_(f.get_tensor("weight").float())

        def forward(self, x):
            normalized = round_torch(self.norm(last_torch(x)))
            return normalized if host_readout else self.linear(normalized)

    model = Readout().eval()
    length = torch.export.Dim("length", min=2, max=config.max_position_embeddings)
    program = torch.export.export(
        model, (torch.from_numpy(hidden),), dynamic_shapes=({1: length},), strict=False
    )

    def convert(node, importer):
        x = importer.env[node.args[0]]
        func = importer.block_builder.add_func(round_bf16, "round_bf16")
        return importer.block_builder.emit(
            relax.call_tir(func, (x,), relax.TensorType((1, 1536), "float32"))
        )

    def convert_last(node, importer):
        x = importer.env[node.args[0]]
        func = importer.block_builder.add_func(last_token, "last_token")
        return importer.block_builder.emit(
            relax.call_tir(func, (x,), relax.TensorType((1, 1536), "float32"))
        )

    module = from_exported_program(
        program,
        unwrap_unit_return_tuple=True,
        keep_params_as_input=True,
        custom_convert_map={
            "bf16_round.default": convert,
            "last_token.default": convert_last,
        },
    )
    module, params = relax.frontend.detach_params(module)
    module, packed = pack_parameters(module, params["main"], 1)
    print(
        "Readout parameters:",
        [name for name, _ in program.named_parameters()],
        flush=True,
    )
    if host_readout:
        module["accumulate"] = accumulate_readout.with_attr(
            "global_symbol", "accumulate"
        )
    return module, packed, model
