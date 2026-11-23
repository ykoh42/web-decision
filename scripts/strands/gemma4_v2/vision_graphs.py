"""Build the independently validated Gemma 4 vision Relax graphs in memory.

Each builder returns an IRModule with a `main` entry point and its detached
parameters. The caller controls linking, lowering, compilation and deployment.
"""

import hashlib
import json
from pathlib import Path

import torch
import transformers
import tvm
from safetensors import safe_open
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4ClippableLinear,
    Gemma4MultimodalEmbedder,
    Gemma4VisionEncoderLayer,
    Gemma4VisionPatchEmbedder,
)
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
from vision_attention_tir import make_parallel_gpu_prim_func
from vision_pool_ir import Pool

MANIFEST = (
    Path(__file__).resolve().parents[3]
    / "packages/web-decision/assets/strands-gemma4-v2/vision-bindings.json"
)
MANIFEST_SHA256 = "23255ad23212dc2d286ed58d2a09deed429b0aac2895536699a4ddb41be2616f"


def load_source(module, checkpoint, prefix):
    names = set(module.state_dict())
    with safe_open(Path(checkpoint) / "model.safetensors", framework="pt") as source:
        weights = {name: source.get_tensor(prefix + name).float() for name in names}
    module.load_state_dict(weights, strict=True)


def check_parameters(parameters, expected):
    if len(parameters) != len(expected):
        raise ValueError(
            "Vision graph parameter count differs from the verified bindings"
        )
    for value, binding in zip(parameters, expected, strict=True):
        array = value.numpy()
        if (
            list(array.shape) != binding["shape"]
            or hashlib.sha256(array.tobytes()).hexdigest() != binding["float32Sha256"]
        ):
            raise ValueError(
                f"Vision graph parameter order or value differs: {binding['name']}"
            )


def import_graph(program, expected, exported_prefix, source_prefix, custom=None):
    names = [
        name.removeprefix(exported_prefix) for name, _ in program.named_parameters()
    ]
    if names != [item["name"].removeprefix(source_prefix) for item in expected]:
        raise ValueError(
            "Vision graph parameter names differ from the verified bindings"
        )
    module = from_exported_program(
        program,
        keep_params_as_input=True,
        unwrap_unit_return_tuple=True,
        custom_convert_map=custom or {},
    )
    module, parameters = relax.frontend.detach_params(module)
    check_parameters(parameters["main"], expected)
    return module, parameters["main"]


def manifest():
    data = MANIFEST.read_bytes()
    if hashlib.sha256(data).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Vision binding manifest digest differs")
    return json.loads(data)


class Patch(torch.nn.Module):
    def __init__(self, patch):
        super().__init__()
        self.patch = patch

    def forward(self, pixels, positions):
        clamped = torch.maximum(positions, torch.zeros_like(positions))
        pos_x, pos_y = clamped.split(1, dim=-1)
        table_x, table_y = self.patch.position_embedding_table.split(1, dim=0)
        x = torch.nn.functional.embedding(pos_x.squeeze(-1), table_x.squeeze(0))
        y = torch.nn.functional.embedding(pos_y.squeeze(-1), table_y.squeeze(0))
        nonnegative = torch.minimum(
            torch.maximum(positions + 1, torch.zeros_like(positions)),
            torch.ones_like(positions),
        )
        valid_x, valid_y = nonnegative.split(1, dim=-1)
        valid = torch.maximum(valid_x, valid_y).squeeze(-1)
        return (
            self.patch.input_proj(2 * (pixels - 0.5))
            + (x + y) * valid.unsqueeze(-1).float()
        )


def build_patch_ir(checkpoint):
    config = transformers.AutoConfig.from_pretrained(str(checkpoint)).vision_config
    patch = Gemma4VisionPatchEmbedder(config).eval()
    load_source(patch, checkpoint, "model.vision_tower.patch_embedder.")
    inputs = (torch.zeros((1, 2520, 768)), torch.zeros((1, 2520, 2), dtype=torch.int32))
    program = torch.export.export(Patch(patch).eval(), inputs, strict=False)
    return import_graph(
        program, manifest()["patch"], "patch.", "model.vision_tower.patch_embedder."
    )


@torch.library.custom_op("webdecision::vision_attention", mutates_args=())
def tiled_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    scores = query.float() @ key.float().transpose(-1, -2)
    scores = scores.masked_fill(valid[:, None, None, :] == 0, -1e30)
    return (torch.softmax(scores, dim=-1) @ value.float()).transpose(1, 2).contiguous()


@tiled_attention.register_fake
def tiled_attention_fake(query, key, value, valid):
    return torch.empty(
        (query.shape[0], query.shape[2], query.shape[1], query.shape[3]),
        dtype=query.dtype,
        device=query.device,
    )


def attention(_module, query, key, value, mask, scaling=1.0, **_kwargs):
    if scaling != 1.0:
        raise ValueError("Expected the pinned Gemma 4 vision attention scale")
    valid = 1 + (mask.reshape(1, -1) * 1e-30).to(torch.int32)
    return tiled_attention(query, key, value, valid), None


class ClippedLinear(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        self.linear = original.linear
        for name in ("input_min", "input_max", "output_min", "output_max"):
            self.register_parameter(
                name,
                torch.nn.Parameter(
                    getattr(original, name).detach().clone(), requires_grad=False
                ),
            )

    def forward(self, values):
        values = torch.minimum(torch.maximum(values, self.input_min), self.input_max)
        values = self.linear(values)
        return torch.minimum(torch.maximum(values, self.output_min), self.output_max)


def lower_clipped_linears(module):
    for name, child in list(module.named_children()):
        if isinstance(child, Gemma4ClippableLinear) and child.use_clipped_linears:
            setattr(module, name, ClippedLinear(child))
        else:
            lower_clipped_linears(child)


class Encoder(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, hidden, cosine, sine, positions, bias):
        return self.block(
            hidden,
            position_embeddings=(cosine, sine),
            position_ids=positions,
            attention_mask=bias,
        )


def convert_attention(node, importer):
    query, key, value, valid = (importer.env[arg] for arg in node.args)
    query = importer.block_builder.emit(relax.op.permute_dims(query, [0, 2, 1, 3]))
    key = importer.block_builder.emit(relax.op.permute_dims(key, [0, 2, 1, 3]))
    value = importer.block_builder.emit(relax.op.permute_dims(value, [0, 2, 1, 3]))
    shape = query.ty.shape.values
    split = importer.block_builder.emit(
        relax.op.full((1,), relax.const(0, "int32"), "int32")
    )
    kernel = importer.block_builder.add_func(
        make_parallel_gpu_prim_func(
            64, None, max_shared_storage_bytes=16384, use_ping_pong=True
        ).with_attr("tirx.is_scheduled", True),
        "vision_attention",
    )
    return importer.block_builder.emit(
        relax.call_tir(
            kernel,
            (query, key, value, valid, split),
            relax.TensorType(tuple(shape), "float32"),
        )
    )


def build_encoder_ir(checkpoint):
    config = transformers.AutoConfig.from_pretrained(str(checkpoint)).vision_config
    transformers.AttentionInterface.register("web_decision_vision_probe", attention)
    config._attn_implementation = "web_decision_vision_probe"
    layer = Gemma4VisionEncoderLayer(config, 0).eval()
    load_source(layer, checkpoint, "model.vision_tower.encoder.layers.0.")
    lower_clipped_linears(layer)
    inputs = (
        torch.zeros((1, 2520, 768)),
        torch.zeros((1, 2520, 64)),
        torch.zeros((1, 2520, 64)),
        torch.zeros((1, 2520, 2), dtype=torch.int32),
        torch.zeros((1, 1, 1, 2520)),
    )
    program = torch.export.export(Encoder(layer).eval(), inputs, strict=False)
    return import_graph(
        program,
        manifest()["layers"][0]["parameters"],
        "block.",
        "model.vision_tower.encoder.layers.0.",
        {"vision_attention.default": convert_attention},
    )


def build_pool_ir():
    n = tvm.tirx.Var("n", "int64")
    builder = relax.BlockBuilder(Pool)
    hidden = relax.Var("hidden", relax.TensorType((1, 2520, 768), "float32"))
    indices = relax.Var("indices", relax.TensorType((n, 9), "int32"))
    with builder.function(
        "main",
        [hidden, indices],
        attrs={"tir_var_lower_bound": {"n": 1}, "tir_var_upper_bound": {"n": 280}},
    ):
        with builder.dataflow():
            result = builder.emit(
                relax.call_tir(
                    builder.get().get_global_var("pool"),
                    (hidden, indices),
                    relax.TensorType((n, 768), "float32"),
                )
            )
            result = builder.emit_output(result)
        builder.emit_func_output(result)
    return builder.get(), []


def build_projection_ir(checkpoint):
    config = transformers.AutoConfig.from_pretrained(str(checkpoint))
    model = Gemma4MultimodalEmbedder(config.vision_config, config.text_config).eval()
    load_source(model, checkpoint, "model.embed_vision.")
    tokens = torch.export.Dim("tokens", min=1, max=280)
    program = torch.export.export(
        model, (torch.zeros((2, 768)),), dynamic_shapes=({0: tokens},)
    )
    return import_graph(program, manifest()["projection"], "", "model.embed_vision.")
