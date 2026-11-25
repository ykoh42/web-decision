"""Direct TVM import for Gemma 4 E2B v1's causal prefill layers."""

from hashlib import file_digest
from pathlib import Path

import torch
import tvm
from bf16_lowering import lower_bf16_program
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer
from tvm import relax

LAYER_INDEX = 0
MAX_LENGTH = 4096
BASE_FILES = {
    "config.json": "1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330",
    "model.safetensors": "2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550",
}
ADAPTER_FILES = {
    "adapter_config.json": "0b9bf880b630441245e20eed71afb8a243cee13d7f0b7022c9e0d77bb267b7bc",
    "adapter_model.safetensors": "c7786a59354aa190082b1c3aedb7ff3311fe260f070f360c4d3518762691c551",
}


def verify_files(directory, pins):
    for name, expected in pins.items():
        with (directory / name).open("rb") as file:
            if file_digest(file, "sha256").hexdigest() != expected:
                raise ValueError(f"Gemma 4 E2B v1 source SHA-256 mismatch for {name}")


class CausalLayer(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        attention = layer.get_base_model().self_attn
        self.layer_type = attention.layer_type
        self.reads_shared_kv = attention.is_kv_shared_layer
        self.writes_shared_kv = attention.store_full_length_kv

    def forward(
        self, hidden, per_layer, cosine, sine, causal_mask, key=None, value=None
    ):
        shared = {self.layer_type: (key, value)} if self.reads_shared_kv else {}
        result = self.layer(
            hidden,
            per_layer,
            attention_mask=causal_mask,
            position_embeddings=(cosine, sine),
            shared_kv_states=shared,
        )
        if self.writes_shared_kv:
            return result, *shared[self.layer_type]
        return result


def load_base_layer(directory, config, index):
    layer = Gemma4TextDecoderLayer(config, index).bfloat16().eval()
    prefix = f"model.language_model.layers.{index}."
    with safe_open(
        directory / "model.safetensors", framework="pt", device="cpu"
    ) as source:
        weights = {
            name: source.get_tensor(prefix + name) for name in layer.state_dict()
        }
    if any(weight.dtype != torch.bfloat16 for weight in weights.values()):
        raise ValueError("The pinned Gemma base layer must contain BF16 tensors")
    layer.load_state_dict(weights, strict=True)
    return layer


def attach_adapter(layer, directory, index):
    config = LoraConfig.from_pretrained(directory, local_files_only=True)
    # PeftModel's generic wrapper forwards the decoder layer's tensor arguments.
    config.task_type = None
    model = get_peft_model(layer, config).eval()
    prefix = f"base_model.model.layers.{index}."
    with safe_open(
        directory / "adapter_model.safetensors", framework="pt", device="cpu"
    ) as source:
        weights = {
            "base_model.model." + name.removeprefix(prefix): source.get_tensor(name)
            for name in source.keys()
            if name.startswith(prefix)
        }
    expected = 10 if layer.self_attn.is_kv_shared_layer else 14
    if len(weights) != expected or any(
        weight.dtype != torch.float32 for weight in weights.values()
    ):
        raise ValueError(f"The pinned layer must contain {expected} FP32 LoRA tensors")
    result = set_peft_model_state_dict(model, weights)
    if result.unexpected_keys or any("lora_" in name for name in result.missing_keys):
        raise ValueError("The Gemma layer did not load every pinned LoRA tensor")
    return model


class CausalLayerSource:
    def __init__(self, base: Path, checkpoint: Path):
        self.base = base
        self.adapter = checkpoint / "lora"
        verify_files(base, BASE_FILES)
        verify_files(self.adapter, ADAPTER_FILES)
        self.config = AutoConfig.from_pretrained(
            base, local_files_only=True
        ).text_config
        self.config._attn_implementation = "eager"

    def load(self, index):
        layer = load_base_layer(self.base, self.config, index)
        return CausalLayer(attach_adapter(layer, self.adapter, index)).eval()


def layer_input_shapes(example):
    length = torch.export.Dim("length", min=2, max=MAX_LENGTH)
    shapes = (
        {1: length},
        {1: length},
        {1: length},
        {1: length},
        {2: length, 3: length},
    )
    cache_shapes = tuple({2: length} for _ in example[5:])
    return shapes + cache_shapes


def import_causal_layer(layer, example):
    shapes = layer_input_shapes(example)
    program = torch.export.export(layer, example, dynamic_shapes=shapes, strict=False)
    module = lower_bf16_program(program, example, shapes)
    module, params = relax.frontend.detach_params(module)
    return module, params["main"]


def compile_causal_layer(module, target):
    # Keep the explicit BF16 boundaries and external runtime parameters.
    with tvm.transform.PassContext(disabled_pass=["StaticPlanBlockMemory", "FuseOps"]):
        return relax.build(module, target=target)
