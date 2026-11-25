"""Execute all pinned Gemma v1 layers with native TVM compute on CPU."""

from dataclasses import dataclass

import torch
from causal_layer_ir import MAX_LENGTH
from safetensors import safe_open
from tiled_causal_layer import (
    CpuStage,
    LayerKv,
    TiledCausalLayer,
    from_tvm,
    import_stage,
    import_tiled_layer,
    to_tvm,
)
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4RMSNorm,
    Gemma4TextRotaryEmbedding,
)

TEXT_TENSORS = "model.language_model."


def load_norm(weights, config, width, name):
    norm = Gemma4RMSNorm(width, eps=config.rms_norm_eps).bfloat16().eval()
    norm.load_state_dict({"weight": weights.get_tensor(TEXT_TENSORS + name)})
    return norm


def embed_rows(weights, name, tokens, scale):
    table = weights.get_slice(TEXT_TENSORS + name)
    rows = torch.cat([table[token : token + 1] for token in tokens]).unsqueeze(0)
    return rows * torch.tensor(scale, dtype=torch.bfloat16)


class DecoderEmbeddings:
    def __init__(self, checkpoint, config):
        self.checkpoint = checkpoint
        self.config = config
        with safe_open(checkpoint, framework="pt", device="cpu") as weights:
            self.projection = weights.get_tensor(
                TEXT_TENSORS + "per_layer_model_projection.weight"
            )
            self.norm = load_norm(
                weights,
                config,
                config.hidden_size_per_layer_input,
                "per_layer_projection_norm.weight",
            )

    def __call__(self, tokens):
        config = self.config
        width = config.hidden_size_per_layer_input
        vocab_size = min(config.vocab_size, config.vocab_size_per_layer_input)
        if any(not 0 <= token < vocab_size for token in tokens):
            raise ValueError("Token ID is outside the pinned Gemma vocabulary")
        with safe_open(self.checkpoint, framework="pt", device="cpu") as weights:
            hidden = embed_rows(
                weights, "embed_tokens.weight", tokens, config.hidden_size**0.5
            )
            identity = embed_rows(
                weights, "embed_tokens_per_layer.weight", tokens, width**0.5
            )
        shape = (1, len(tokens), config.num_hidden_layers, width)
        projection = torch.nn.functional.linear(hidden, self.projection)
        projection = (projection * config.hidden_size**-0.5).reshape(shape)
        per_layer = (self.norm(projection) + identity.reshape(shape)) * 2.0**-0.5
        return hidden, per_layer


def compile_layers(source):
    config = source.config
    hidden = torch.zeros((1, 3, config.hidden_size), dtype=torch.bfloat16)
    per_layer = torch.zeros(
        (1, 3, config.hidden_size_per_layer_input), dtype=torch.bfloat16
    )
    positions = torch.arange(3).unsqueeze(0)
    rotary = Gemma4TextRotaryEmbedding(config)
    for index in range(config.num_hidden_layers):
        layer = source.load(index)
        cosine, sine = rotary(hidden, positions, layer_type=layer.layer_type)
        programs = import_tiled_layer(layer, (hidden, per_layer, cosine, sine))
        yield TiledCausalLayer(layer, programs)


def shared_producers(layers):
    producers = {}
    consumers = {}
    for index, layer in enumerate(layers):
        if layer.reads_shared:
            if layer.layer_type not in producers:
                raise ValueError(f"Gemma layer {index} has no earlier shared K/V")
            consumers[index] = producers[layer.layer_type]
        if layer.writes_shared:
            producers[layer.layer_type] = index
    return consumers


@dataclass(frozen=True)
class DecoderPrefix:
    layers: tuple[LayerKv | None, ...]
    end: int


class TiledCausalDecoder:
    """CPU runner for the verified text checkpoint and its reusable K/V."""

    def __init__(self, source):
        config = source.config
        checkpoint = source.base / "model.safetensors"
        self.layers = tuple(compile_layers(source))
        self.shared_producers = shared_producers(self.layers)
        self.embeddings = DecoderEmbeddings(checkpoint, config)
        with safe_open(checkpoint, framework="pt", device="cpu") as weights:
            norm = load_norm(weights, config, config.hidden_size, "norm.weight")
        example = torch.zeros((1, 3, config.hidden_size), dtype=torch.bfloat16)
        self.final_norm = CpuStage(import_stage(norm, (example,)))

    @torch.inference_mode()
    def __call__(self, tokens, *, prefix=None):
        start = prefix.end if prefix is not None else 0
        end = start + len(tokens)
        if not 0 <= start < end <= MAX_LENGTH:
            raise ValueError("The decoder positions must fit the pinned context")
        previous = prefix.layers if prefix is not None else (None,) * len(self.layers)
        hidden, per_layer = self.embeddings(tokens)
        next_prefix = []
        shared = {}
        for index, layer in enumerate(self.layers):
            producer = self.shared_producers.get(index)
            hidden, cached, active = layer(
                hidden,
                per_layer[:, :, index],
                prefix=previous[index],
                shared=shared.get(producer),
            )
            next_prefix.append(cached)
            if active is not None:
                shared[index] = active
        return from_tvm(self.final_norm(to_tvm(hidden))), DecoderPrefix(
            tuple(next_prefix), end
        )
