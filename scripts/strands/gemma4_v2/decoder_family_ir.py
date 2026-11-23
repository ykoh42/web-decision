"""Build one packed Gemma 4 text decoder graph family from the pinned checkpoint."""

import torch
import transformers
from packed_parameters import pack_parameters
from prefix_attention_chunked import make_parallel_gpu_prim_func
from safetensors import safe_open
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program


@torch.library.custom_op("webdecision::prefix_attention", mutates_args=())
def tiled_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    split: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    length = query.shape[2]
    index = torch.arange(length, device=query.device)
    blocked = ((index[:, None] < split[0]) & (index[None, :] >= split[0])) | (
        valid[0][None, :] == 0
    )
    scores = query.float() @ key.float().transpose(-1, -2)
    scores = scores.masked_fill(blocked, -1e30)
    return (torch.softmax(scores, dim=-1) @ value.float()).transpose(1, 2).contiguous()


@tiled_attention.register_fake
def tiled_attention_fake(query, key, value, split, valid):
    return torch.empty(
        (query.shape[0], query.shape[2], query.shape[1], query.shape[3]),
        dtype=query.dtype,
        device=query.device,
    )


def load_layer(checkpoint, index):
    config = transformers.AutoConfig.from_pretrained(str(checkpoint)).text_config

    def attention(_module, query, key, value, mask, **_kwargs):
        return tiled_attention(query, key, value, mask[0], mask[1]), None

    transformers.AttentionInterface.register("web_decision_dynamic_probe", attention)
    config._attn_implementation = "web_decision_dynamic_probe"
    layer = Gemma4TextDecoderLayer(config, index).eval()
    with safe_open(checkpoint / "model.safetensors", framework="pt") as source:
        prefix = f"model.language_model.layers.{index}."
        names = set(layer.state_dict())
        weights = {
            key[len(prefix) :]: source.get_tensor(key).float()
            for key in source.keys()
            if key.startswith(prefix) and key[len(prefix) :] in names
        }
    layer.load_state_dict(weights, strict=True)
    scalar = layer.layer_scalar.detach().float().clone()
    layer.layer_scalar.fill_(1.0)
    shared = layer.self_attn.is_kv_shared_layer
    if not shared:
        layer.self_attn.store_full_length_kv = True
    return config, layer, scalar, shared


class DynamicLayer(torch.nn.Module):
    def __init__(self, block, shared):
        super().__init__()
        self.block = block
        self.shared = shared

    def forward(self, hidden, per_layer, cosine, sine, mask, valid, scale, *cache):
        shared_kv = (
            {self.block.self_attn.layer_type: (cache[0], cache[1])} if cache else {}
        )
        output = self.block(
            hidden,
            per_layer,
            shared_kv_states=shared_kv,
            position_embeddings=(cosine, sine),
            attention_mask=(mask, valid),
        )
        output = output * scale
        if not self.shared:
            key, value = shared_kv[self.block.self_attn.layer_type]
            return output, key, value
        return output


def convert_attention(node, importer):
    query, key, value, split, valid = (importer.env[arg] for arg in node.args)
    query = importer.block_builder.emit(relax.op.permute_dims(query, [0, 2, 1, 3]))
    key = importer.block_builder.emit(relax.op.permute_dims(key, [0, 2, 1, 3]))
    value = importer.block_builder.emit(relax.op.permute_dims(value, [0, 2, 1, 3]))
    shape = query.ty.shape.values
    kernel = importer.block_builder.add_func(
        make_parallel_gpu_prim_func(
            int(shape[3]), None, max_shared_storage_bytes=16384
        ).with_attr("tirx.is_scheduled", True),
        "prefix_attention",
    )
    return importer.block_builder.emit(
        relax.call_tir(
            kernel,
            (query, key, value, valid, split),
            relax.TensorType(tuple(shape), "float32"),
        )
    )


def build_decoder_family(checkpoint, index, example):
    config, layer, scalar, shared = load_layer(checkpoint, index)
    length = torch.export.Dim("length", min=2, max=config.max_position_embeddings)
    dynamic = (
        {1: length},
        {1: length},
        {1: length},
        {1: length},
        None,
        {1: length},
        None,
        *((({2: length}, {2: length}),) if shared else ()),
    )
    inputs = (
        example["hidden"],
        example["per_layer_input"],
        *example["position_embeddings"],
        torch.tensor([24], dtype=torch.int32),
        torch.ones((1, 80), dtype=torch.int32),
        scalar,
        *(example["shared_kv"][layer.self_attn.layer_type] if shared else ()),
    )
    program = torch.export.export(
        DynamicLayer(layer, shared).eval(), inputs, dynamic_shapes=dynamic, strict=False
    )
    module = from_exported_program(
        program,
        unwrap_unit_return_tuple=True,
        keep_params_as_input=True,
        custom_convert_map={"prefix_attention.default": convert_attention},
    )
    module, params = relax.frontend.detach_params(module)
    module, _ = pack_parameters(module, params["main"], 9 if shared else 7)
    return module
