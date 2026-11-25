"""Generate a tiny actual E2B v1 BF16 pointer reference without WebGPU."""

import argparse
import json
from pathlib import Path

import torch
from causal_layer_ir import ADAPTER_FILES, BASE_FILES, verify_files
from pointer_head_ir import HEAD_SHA256
from source_inventory import BASE_REVISION
from strands_decider.modeling import StrandsDeciderModel


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", type=Path)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    torch.set_num_threads(2)
    verify_files(args.base, BASE_FILES)
    verify_files(args.checkpoint / "lora", ADAPTER_FILES)
    verify_files(args.checkpoint, {"head.safetensors": HEAD_SHA256})
    model = StrandsDeciderModel.load(
        str(args.checkpoint), device_map="cpu", attn_implementation="eager"
    ).eval()
    if (
        model.config.base_revision != BASE_REVISION
        or next(model.torso.parameters()).dtype != torch.bfloat16
    ):
        raise ValueError("Unexpected original model or dtype")
    state = [2]
    question = {"ids": [50, 120, 230], "optionIndices": [0, 1], "answerIndex": 2}
    tokens = torch.tensor([state + question["ids"]])
    hidden = model.torso(
        input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False
    ).last_hidden_state
    logits = model.head(hidden[:, -1].float(), hidden[:, [1, 2]].float())[0].tolist()
    with args.out.open("x") as output:
        json.dump(
            {
                "modelSha": BASE_FILES["model.safetensors"],
                "adapterSha": ADAPTER_FILES["adapter_model.safetensors"],
                "state": state,
                "question": question,
                "logits": logits,
            },
            output,
        )
        output.write("\n")


if __name__ == "__main__":
    main()
