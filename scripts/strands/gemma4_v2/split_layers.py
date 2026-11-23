"""Attention and feed-forward sections of a Gemma decoder layer."""

import torch


class AttentionPart(torch.nn.Module):
    def __init__(self, block, shared):
        super().__init__()
        self.self_attn = block.self_attn
        self.input_layernorm = block.input_layernorm
        self.post_attention_layernorm = block.post_attention_layernorm
        self.shared = shared

    def forward(self, hidden, cosine, sine, mask, valid, *cache):
        shared_kv = {self.self_attn.layer_type: (cache[0], cache[1])} if cache else {}
        residual = hidden
        hidden = self.input_layernorm(hidden)
        hidden, _ = self.self_attn(
            hidden_states=hidden,
            position_embeddings=(cosine, sine),
            attention_mask=(mask, valid),
            shared_kv_states=shared_kv,
        )
        hidden = residual + self.post_attention_layernorm(hidden)
        if not self.shared:
            key, value = shared_kv[self.self_attn.layer_type]
            return hidden, key, value
        return hidden


class Tail(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.pre_feedforward_layernorm = block.pre_feedforward_layernorm
        self.mlp = block.mlp
        self.post_feedforward_layernorm = block.post_feedforward_layernorm
        self.per_layer_input_gate = block.per_layer_input_gate
        self.act_fn = block.act_fn
        self.per_layer_projection = block.per_layer_projection
        self.post_per_layer_input_norm = block.post_per_layer_input_norm

    def forward(self, hidden, per_layer, scale):
        residual = hidden
        hidden = (
            self.post_feedforward_layernorm(
                self.mlp(self.pre_feedforward_layernorm(hidden))
            )
            + residual
        )
        residual = hidden
        hidden = self.per_layer_input_gate(hidden)
        hidden = self.act_fn(hidden) * per_layer
        hidden = (
            self.post_per_layer_input_norm(self.per_layer_projection(hidden)) + residual
        )
        return hidden * scale
