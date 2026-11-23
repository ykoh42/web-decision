"""Scratch TVM query-chunk graph families using the existing checkpoint ABI."""

import math

import torch
from decoder_family_ir import load_layer
from model_partition import partition_model
from packed_parameters import make_unpack
from query_attention_tir import make_parallel_gpu_prim_func
from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

QUERY_WEIGHTS = (
    "self_attn.q_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
)
KV_WEIGHTS = (
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.k_norm.weight",
)
UNSHARED_PACKED = (
    QUERY_WEIGHTS[0],
    QUERY_WEIGHTS[2],
    KV_WEIGHTS[2],
    KV_WEIGHTS[0],
    KV_WEIGHTS[1],
    QUERY_WEIGHTS[1],
    *QUERY_WEIGHTS[3:],
)
SHARED_PACKED = (
    QUERY_WEIGHTS[0],
    QUERY_WEIGHTS[2],
    QUERY_WEIGHTS[1],
    *QUERY_WEIGHTS[3:],
)


@torch.library.custom_op("webdecision_query_scratch::attention", mutates_args=())
def query_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid: torch.Tensor,
    split_and_start: torch.Tensor,
) -> torch.Tensor:
    qp = torch.arange(query.shape[1], device=query.device) + split_and_start[1]
    kp = torch.arange(key.shape[2], device=query.device)
    allowed = valid[0][None, :].bool() & (
        (qp[:, None] >= split_and_start[0]) | (kp[None, :] < split_and_start[0])
    )
    scores = query.transpose(1, 2) @ key.transpose(-1, -2)
    weights = torch.softmax(scores.masked_fill(~allowed, -1e30), dim=-1)
    weights = torch.where(allowed.any(dim=-1)[:, None], weights, 0)
    return (weights @ value).transpose(1, 2).contiguous()


@query_attention.register_fake
def fake_attention(query, key, value, valid, split_and_start):
    return torch.empty_like(query)


class QueryPart(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.q_norm = block.self_attn.q_norm
        self.input_layernorm = block.input_layernorm
        self.post_attention_layernorm = block.post_attention_layernorm
        self.width = block.self_attn.head_dim

    def forward(
        self, hidden, cosine, sine, valid, split_and_start, key, value, q_t, o_t
    ):
        normalized = self.input_layernorm(hidden)
        query = (normalized @ q_t).reshape(1, hidden.shape[1], 8, self.width)
        query = apply_rotary_pos_emb(self.q_norm(query), cosine, sine, unsqueeze_dim=2)
        attended = query_attention(query, key, value, valid, split_and_start)
        projected = attended.reshape(1, hidden.shape[1], 8 * self.width) @ o_t
        return hidden + self.post_attention_layernorm(projected)


class KvPart(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.k_proj = block.self_attn.k_proj
        self.v_proj = block.self_attn.v_proj
        self.k_norm = block.self_attn.k_norm
        self.v_norm = block.self_attn.v_norm
        self.input_layernorm = block.input_layernorm
        self.width = block.self_attn.head_dim

    def forward(self, hidden, cosine, sine):
        normalized = self.input_layernorm(hidden)
        shape = (1, hidden.shape[1], 1, self.width)
        k = self.k_proj(normalized).reshape(shape)
        v = self.v_proj(normalized).reshape(shape) if self.v_proj is not None else k
        key = apply_rotary_pos_emb(self.k_norm(k), cosine, sine, unsqueeze_dim=2)
        return key.transpose(1, 2).contiguous(), self.v_norm(v).transpose(
            1, 2
        ).contiguous()


def convert_attention(node, importer):
    query, key, value, valid, metadata = (importer.env[arg] for arg in node.args)
    shape = query.ty.shape.values
    function = importer.block_builder.add_func(
        make_parallel_gpu_prim_func(
            int(shape[-1]),
            None,
            max_shared_storage_bytes=16384,
            separate_query_rows=True,
        ).with_attr("tirx.is_scheduled", True),
        "prefix_attention",
    )
    return importer.block_builder.emit(
        relax.call_tir(
            function,
            (query, key, value, valid, metadata),
            relax.TensorType(tuple(shape), "float32"),
        )
    )


def import_part(part, args, dynamic, names, converters=None):
    exported = torch.export.export(
        part.eval(), args, dynamic_shapes=dynamic, strict=False
    )
    actual = tuple(name for name, _ in exported.named_parameters())
    if actual != names:
        raise ValueError(f"Unexpected query graph weight order: {actual}")
    module = from_exported_program(
        exported,
        unwrap_unit_return_tuple=True,
        keep_params_as_input=True,
        custom_convert_map=converters,
    )
    module, _ = relax.frontend.detach_params(module)
    return partition_model(module, "main", preserve_tir=True)


def build_query(block, width):
    rows = torch.export.Dim("query_rows", min=1, max=256)
    keys = torch.export.Dim("key_rows", min=2, max=131072)
    return import_part(
        QueryPart(block),
        (
            torch.zeros((1, 3, 1536)),
            torch.ones((1, 3, width)),
            torch.zeros((1, 3, width)),
            torch.ones((1, 9), dtype=torch.int32),
            torch.tensor([4, 0], dtype=torch.int32),
            torch.zeros((1, 1, 9, width)),
            torch.zeros((1, 1, 9, width)),
            torch.empty((1536, 8 * width)),
            torch.empty((8 * width, 1536)),
        ),
        (
            {1: rows},
            {1: rows},
            {1: rows},
            {1: keys},
            None,
            {2: keys},
            {2: keys},
            None,
            None,
        ),
        ("q_norm.weight", "input_layernorm.weight", "post_attention_layernorm.weight"),
        {"attention.default": convert_attention},
    )


def build_kv(block, width):
    keys = torch.export.Dim("key_rows", min=2, max=131072)
    return import_part(
        KvPart(block),
        (
            torch.zeros((1, 9, 1536)),
            torch.ones((1, 9, width)),
            torch.zeros((1, 9, width)),
        ),
        ({1: keys}, {1: keys}, {1: keys}),
        ("k_proj.weight", "v_proj.weight", "k_norm.weight", "input_layernorm.weight"),
    )


def prepare_weights(block, shared):
    builder = relax.BlockBuilder()
    packed_names = SHARED_PACKED if shared else UNSHARED_PACKED
    shapes = {name: tuple(block.state_dict()[name].shape) for name in packed_names}
    symbols = {
        name: builder.add_func(make_unpack(shapes[name]), f"unpack_{index}")
        for index, name in enumerate(packed_names)
    }
    inputs = [
        relax.Var(
            f"packed_{index}",
            relax.TensorType((math.prod(shapes[name]) // 2,), "uint32"),
        )
        for index, name in enumerate(packed_names)
    ]
    with builder.function("main", inputs):
        with builder.dataflow():
            decoded = {}
            for var, name in zip(inputs, packed_names, strict=True):
                value = builder.emit(
                    relax.call_tir(
                        symbols[name], (var,), relax.TensorType(shapes[name], "float32")
                    )
                )
                if name in QUERY_WEIGHTS[:2]:
                    value = builder.emit(relax.op.permute_dims(value, [1, 0]))
                decoded[name] = value
            wanted = QUERY_WEIGHTS if shared else (*QUERY_WEIGHTS, *KV_WEIGHTS)
            out = builder.emit_output(relax.Tuple([decoded[name] for name in wanted]))
        builder.emit_func_output(out)
    return builder.get()


def build(checkpoint, index):
    _, block, _, shared = load_layer(checkpoint, index)
    width = block.self_attn.head_dim
    result = [
        (f"query{index}_chunk", build_query(block, width)),
        (f"query{index}_weights", prepare_weights(block, shared)),
    ]
    if not shared:
        result.append((f"query{index}_kv", build_kv(block, width)))
    return result
