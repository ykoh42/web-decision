"""Capture real text-layer attention tensors and outputs for a block-attention probe."""

import argparse
from pathlib import Path

import torch
import transformers
from safetensors import safe_open
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    fixture = torch.load(args.capture, weights_only=True)
    split = fixture["state_length"]
    config = transformers.AutoConfig.from_pretrained(str(args.checkpoint)).text_config

    def attention(_module, query, key, value, _mask, **_kwargs):
        scores = query.float() @ key.float().transpose(-1, -2)
        state = (
            torch.softmax(scores[:, :, :split, :split], dim=-1)
            @ value[:, :, :split].float()
        )
        question = torch.softmax(scores[:, :, split:, :], dim=-1) @ value.float()
        output = torch.cat((state, question), dim=2).to(query.dtype).transpose(1, 2)
        torch.save(
            {
                "query": query.float().cpu(),
                "key": key.float().cpu(),
                "value": value.float().cpu(),
                "context": output.float().cpu(),
                "split": split,
            },
            args.output,
        )
        return output, None

    transformers.AttentionInterface.register(
        "web_decision_capture_attention", attention
    )
    config._attn_implementation = "web_decision_capture_attention"
    layer = Gemma4TextDecoderLayer(config, 0).eval()
    prefix = "model.language_model.layers.0."
    with safe_open(args.checkpoint / "model.safetensors", framework="pt") as file:
        names = set(layer.state_dict())
        values = {
            key[len(prefix) :]: file.get_tensor(key).float()
            for key in file.keys()
            if key.startswith(prefix) and key[len(prefix) :] in names
        }
    layer.load_state_dict(values, strict=True)
    layer.to("mps")
    with torch.inference_mode():
        output = layer(
            fixture["hidden"].to("mps"),
            fixture["per_layer_input"].to("mps"),
            shared_kv_states={},
            position_embeddings=tuple(
                value.to("mps") for value in fixture["position_embeddings"]
            ),
            attention_mask=None,
        )
    result = torch.load(args.output, weights_only=True)
    result["layer_output"] = output.float().cpu()
    torch.save(result, args.output)
    print("Captured actual layer Q/K/V and reference attention", flush=True)


if __name__ == "__main__":
    main()
