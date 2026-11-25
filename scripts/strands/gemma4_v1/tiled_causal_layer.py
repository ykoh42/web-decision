"""Run the Gemma v1 layer with row-tiled causal attention and a prefix cache."""

from dataclasses import dataclass

import torch
import tvm
from bf16_lowering import lower_bf16_program
from causal_attention_ir import MAX_QUERY_TILE, import_attention_tile, query_tiles
from causal_layer_ir import MAX_LENGTH, compile_causal_layer
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4TextRotaryEmbedding,
    apply_rotary_pos_emb,
)
from tvm import relax


class ProjectQkv(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.norm = block.input_layernorm
        self.attention = block.self_attn

    def forward(self, hidden, cosine, sine):
        hidden = self.norm(hidden)
        args = self.attention
        shape = (*hidden.shape[:-1], -1, args.head_dim)
        query = args.q_norm(args.q_proj(hidden).reshape(shape))
        query = apply_rotary_pos_emb(query, cosine, sine, unsqueeze_dim=2).transpose(
            1, 2
        )
        if args.is_kv_shared_layer:
            return query
        key = args.k_norm(args.k_proj(hidden).reshape(shape))
        key = apply_rotary_pos_emb(key, cosine, sine, unsqueeze_dim=2).transpose(1, 2)
        value = args.v_norm(args.v_proj(hidden).reshape(shape)).transpose(1, 2)
        return query, key, value


class AttentionTail(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, hidden, per_layer, attended):
        block = self.block
        attended = block.self_attn.o_proj(attended.reshape(*hidden.shape[:-1], -1))
        hidden = hidden + block.post_attention_layernorm(attended)
        residual = block.mlp(block.pre_feedforward_layernorm(hidden))
        hidden = hidden + block.post_feedforward_layernorm(residual)
        residual = block.act_fn(block.per_layer_input_gate(hidden)) * per_layer
        residual = block.post_per_layer_input_norm(block.per_layer_projection(residual))
        return (hidden + residual) * block.layer_scalar


def import_stage(stage, inputs, max_length=MAX_LENGTH):
    length = torch.export.Dim("tokens", min=1, max=max_length)
    shapes = tuple({1: length} for _ in inputs)
    program = torch.export.export(
        stage.eval(), inputs, dynamic_shapes=shapes, strict=False
    )
    module = lower_bf16_program(program, inputs, shapes)
    module, params = relax.frontend.detach_params(module)
    return module, params["main"]


def import_tiled_layer(layer, example, max_keys=MAX_LENGTH):
    hidden, per_layer, cosine, sine = example
    block = layer.layer.get_base_model()
    width = block.self_attn.head_dim
    query = torch.zeros((1, 8, 3, width), dtype=torch.bfloat16)
    key = torch.zeros((1, 1, 5, width), dtype=torch.bfloat16)
    mask = torch.zeros((1, 1, 3, 5), dtype=torch.bfloat16)
    attended = torch.zeros((*hidden.shape[:2], 8, width), dtype=torch.bfloat16)
    return (
        import_stage(ProjectQkv(block), (hidden, cosine, sine)),
        (import_attention_tile((query, key, key.clone(), mask), max_keys), ()),
        import_stage(
            AttentionTail(block), (hidden, per_layer, attended), MAX_QUERY_TILE
        ),
    )


@dataclass(frozen=True)
class LayerKv:
    key: torch.Tensor
    value: torch.Tensor
    end: int

    def __post_init__(self):
        if self.key.shape != self.value.shape or self.key.ndim != 4:
            raise ValueError(
                "K/V must have equal batch, head, position and feature shapes"
            )
        if self.key.dtype != torch.bfloat16 or self.value.dtype != torch.bfloat16:
            raise ValueError("K/V storage must retain the original BF16 bits")
        if not 0 <= self.start < self.end <= MAX_LENGTH:
            raise ValueError("The K/V positions must fit the pinned context")

    @property
    def start(self):
        return self.end - self.key.shape[2]

    def compact(self, window):
        if window is None or self.key.shape[2] < window:
            return self
        first = self.key.shape[2] - window + 1
        return LayerKv(
            self.key[:, :, first:].clone(), self.value[:, :, first:].clone(), self.end
        )


def append_kv(prefix, key, value):
    if prefix is None:
        return LayerKv(key, value, key.shape[2])
    return LayerKv(
        torch.cat((prefix.key, key), dim=2),
        torch.cat((prefix.value, value), dim=2),
        prefix.end + key.shape[2],
    )


def tile_mask(query_start, query_count, key_start, key_count, window):
    if not 1 <= query_count <= MAX_QUERY_TILE or not 1 <= key_count <= MAX_LENGTH:
        raise ValueError(
            "The causal mask must fit one query tile and the pinned context"
        )
    queries = torch.arange(query_start, query_start + query_count).view(-1, 1)
    keys = torch.arange(key_start, key_start + key_count).view(1, -1)
    blocked = keys > queries
    if window is not None:
        blocked |= keys <= queries - window
    mask = torch.zeros((1, 1, query_count, key_count), dtype=torch.bfloat16)
    return mask.masked_fill(blocked.view(mask.shape), torch.finfo(torch.bfloat16).min)


def to_tvm(value):
    return tvm.runtime.tensor(value.detach().float().numpy(), device=tvm.cpu())


def from_tvm(value):
    return torch.from_numpy(value.numpy()).bfloat16()


class CpuStage:
    def __init__(self, compiled):
        module, self.weights = compiled
        self.vm = relax.VirtualMachine(compile_causal_layer(module, "llvm"), tvm.cpu())

    def __call__(self, *inputs):
        return self.vm["main"](*inputs, *self.weights)


class TiledCausalLayer:
    """CPU orchestration of the native TVM stages, with compact reusable BF16 K/V."""

    def __init__(self, layer, programs):
        block = layer.layer.get_base_model()
        self.layer_type = block.self_attn.layer_type
        self.window = block.self_attn.sliding_window
        self.reads_shared = layer.reads_shared_kv
        self.writes_shared = layer.writes_shared_kv
        self.rotary = Gemma4TextRotaryEmbedding(block.config)
        self.project, self.attention, self.tail = map(CpuStage, programs)

    def attend(self, query, active, start):
        key, value = to_tvm(active.key), to_tvm(active.value)
        for interval in query_tiles(query.shape[2]):
            mask = tile_mask(
                start + interval.start,
                interval.stop - interval.start,
                active.start,
                active.key.shape[2],
                self.window,
            )
            yield (
                interval,
                self.attention(to_tvm(query[:, :, interval]), key, value, to_tvm(mask)),
            )

    def finish_tiles(self, hidden, per_layer, query, active, start):
        outputs = []
        for interval, attended in self.attend(query, active, start):
            tail = self.tail(
                to_tvm(hidden[:, interval]), to_tvm(per_layer[:, interval]), attended
            )
            outputs.append(from_tvm(tail))
        return torch.cat(outputs, dim=1)

    def first_query_position(self, length, prefix, shared):
        if self.reads_shared:
            if shared is None or prefix is not None:
                raise ValueError("A shared layer requires its producer's active K/V")
            return shared.end - length
        if shared is not None:
            raise ValueError("A K/V producer uses its own prefix")
        return prefix.end if prefix is not None else 0

    def __call__(self, hidden, per_layer, *, prefix=None, shared=None):
        length = hidden.shape[1]
        if hidden.dtype != torch.bfloat16 or per_layer.dtype != torch.bfloat16:
            raise ValueError("Layer inputs must retain the original BF16 bits")
        start = self.first_query_position(length, prefix, shared)
        if length == 0 or start < 0 or start + length > MAX_LENGTH:
            raise ValueError("The layer positions must fit the pinned context")
        positions = torch.arange(start, start + length).unsqueeze(0)
        cosine, sine = self.rotary(hidden, positions, layer_type=self.layer_type)
        projected = self.project(to_tvm(hidden), to_tvm(cosine), to_tvm(sine))
        if self.reads_shared:
            query, active, next_prefix = from_tvm(projected), shared, None
        else:
            query, key, value = map(from_tvm, projected)
            active = append_kv(prefix, key, value)
            next_prefix = active.compact(self.window)
        first_visible = (
            max(0, start - self.window + 1) if self.window is not None else 0
        )
        if active.start != first_visible or active.end != start + length:
            raise ValueError("The K/V input must cover the layer's visible positions")
        result = self.finish_tiles(hidden, per_layer, query, active, start)
        return result, next_prefix, active if self.writes_shared else None
