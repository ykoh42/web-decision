"""Compare pinned eager decoder arithmetic with the TVM text candidate's BF16 boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from reference import (
    REPOSITORY,
    REVISION,
    SMALL_FILES,
    UPSTREAM_REVISION,
    WEIGHT_FILES,
    use_source,
    verify_files,
)


def prepared_row(path, case_id, row_index):
    data = json.loads(path.read_text())
    source = data["source"]
    if (source["repository"], source["revision"], source["deciderRevision"]) != (
        REPOSITORY,
        REVISION,
        UPSTREAM_REVISION,
    ):
        raise ValueError("Use a prepared fixture from the pinned Mapika oracle")
    case = next(item for item in data["cases"] if item["id"] == case_id)
    if case["status"] != "accepted" or case.get("imageGridThw"):
        raise ValueError("Select an accepted text fixture")
    return case["rows"][row_index]


def packed_projection(hidden, weight):
    import torch

    # Mirror the 32 TIR lanes, each processing one packed BF16 weight pair per group.
    products = hidden.float()[:, None, :] * weight.float()[None, :, :]
    pairs = products.reshape(hidden.shape[0], weight.shape[0], 32, 32, 2)
    partial = torch.zeros_like(pairs[:, :, 0, :, 0])
    for group in range(32):
        partial = partial + pairs[:, :, group, :, 0]
        partial = partial + pairs[:, :, group, :, 1]
    for stride in (16, 8, 4, 2, 1):
        partial = partial[:, :, :stride] + partial[:, :, stride : 2 * stride]
    return partial[:, :, 0].to(torch.bfloat16).float()


def metrics(actual, expected):
    delta = (actual.float() - expected.float()).abs()
    return {
        "count": delta.numel(),
        "changed": int((delta != 0).sum()),
        "maxAbs": float(delta.max()),
        "rms": float(delta.square().mean().sqrt()),
    }


def sha256_bf16(tensor):
    import torch

    data = tensor.contiguous().view(torch.uint16).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def trace_hidden(model, row, *, padded, math_sdpa=False, gdn_export=False):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers.models.qwen3_5 import modeling_qwen3_5

    ids = row["inputIds"]
    width = (len(ids) + 63) // 64 * 64 if padded else len(ids)
    inputs = torch.full((1, width), model.tok.pad_token_id, dtype=torch.long)
    inputs[0, : len(ids)] = torch.tensor(ids)
    mask = torch.zeros_like(inputs)
    mask[0, : len(ids)] = 1
    slots = torch.tensor(row["slots"], dtype=torch.long)
    layers = model.lm.model.layers
    trace = {}

    def capture(key):
        def receive(_module, _inputs, output):
            trace[key] = output[0, slots].clone()

        return receive

    with ExitStack() as stack:
        stack.enter_context(torch.inference_mode())
        if math_sdpa:
            stack.enter_context(sdpa_kernel(SDPBackend.MATH))
        if gdn_export:
            stack.enter_context(
                patch.object(modeling_qwen3_5, "is_torchdynamo_exporting", lambda: True)
            )
        for index, layer in enumerate(layers):
            hook = layer.register_forward_hook(capture(f"layer-{index:02}"))
            stack.callback(hook.remove)
        final = model.lm.model(input_ids=inputs, attention_mask=mask).last_hidden_state
        trace["final-norm"] = final[0, slots].clone()
    return trace


def valid_logits(tensor, row):
    return [
        slot[:count].tolist()
        for slot, count in zip(tensor, row["optionCounts"], strict=True)
    ]


def report_variants(model, row):
    import torch
    from torch.nn import functional as F

    letters = model.lm.lm_head.weight[model.letters].detach()
    variants = {
        "padded-source": {"padded": True},
        "unpadded-source": {"padded": False},
        "unpadded-math-sdpa": {"padded": False, "math_sdpa": True},
        "unpadded-gdn-export-formula": {"padded": False, "gdn_export": True},
    }
    reports = {}
    baseline = None
    for name, options in variants.items():
        current = trace_hidden(model, row, **options)
        if baseline is None:
            baseline = current
        final = current["final-norm"]
        with torch.inference_mode():
            native = F.linear(final, letters).float()
            packed = packed_projection(final, letters)
        reports[name] = {
            "validLogits": valid_logits(native, row),
            "packedProjectionValidLogits": valid_logits(packed, row),
            "layers": {
                layer: {
                    "sha256Bf16": sha256_bf16(value),
                    **metrics(value, baseline[layer]),
                }
                for layer, value in current.items()
            },
        }
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-source", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--case", default="typed-packed")
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    use_source(args.upstream_source.resolve(strict=True))
    verify_files(
        args.snapshot,
        {**SMALL_FILES, "model.safetensors": WEIGHT_FILES["model.safetensors"]},
    )
    row = prepared_row(args.prepared, args.case, args.row)

    import torch
    from decider.model import DecisionModel

    torch.set_num_threads(2)
    model = DecisionModel(
        str(args.snapshot), dtype=torch.bfloat16, grad_ckpt=False
    ).eval()
    report = {
        "source": {
            "repository": REPOSITORY,
            "revision": REVISION,
            "deciderRevision": UPSTREAM_REVISION,
        },
        "backend": {
            "torch": torch.__version__,
            "device": "cpu",
            "activation": "bfloat16",
        },
        "case": args.case,
        "row": args.row,
        "options": row["optionCounts"],
        "variants": report_variants(model, row),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
