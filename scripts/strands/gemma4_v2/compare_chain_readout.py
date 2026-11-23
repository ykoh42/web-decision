"""Compare a GPU decoder-chain sample with the official BF16 readout."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from safetensors import safe_open
from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("actual", type=Path)
    args = parser.parse_args()
    config = transformers.AutoConfig.from_pretrained(str(args.checkpoint)).text_config
    hidden = np.fromfile(args.actual, dtype=np.float32).reshape(
        1, -1, config.hidden_size
    )
    norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    with safe_open(args.checkpoint / "model.safetensors", framework="pt") as file:
        norm.weight.data.copy_(
            file.get_tensor("model.language_model.norm.weight").float()
        )
    with torch.inference_mode():
        row = norm(torch.from_numpy(hidden))[0, -1].numpy()
    reference = np.fromfile(args.reference / "hidden.bin", dtype=np.float32).reshape(
        -1, config.hidden_size
    )[0]
    sys.path.insert(0, str(args.upstream / "src"))
    from strands_decider.core.readout import (
        code_probabilities,
        load_readout,
        readout_logits,
    )

    options = json.loads((args.checkpoint / "decision_config.json").read_text())
    weights = load_readout(args.checkpoint)
    result = {}
    for name, candidate in (("official_bf16", reference), ("gpu_fp32_chain", row)):
        logits = readout_logits(weights, candidate)[:2]
        result[name] = code_probabilities(
            logits, options["final_logit_softcapping"], options["temperature"]
        ).tolist()
    result["hidden_max_abs_error"] = float(np.abs(row - reference).max())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
