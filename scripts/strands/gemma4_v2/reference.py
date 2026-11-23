"""Run the pinned upstream Strands E2B v2 engine against local model files."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PINS = {
    "model.safetensors": "593a28a67754653c7ffffff80a5186f18966cc6262c2f4f097414c187e5fd48c",
    "readout.safetensors": "37b3f1c4670f1afb78007624e3ec2a2f55503d9b78602216e2f29a4cfb5111f6",
    "tokenizer.json": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
}


def setup_capture(engine, args):
    if args.capture_dir:
        args.capture_dir.mkdir(parents=True, exist_ok=True)
        forward = engine.backend.forward

        def capture_layer(index, _module, inputs, kwargs, output):
            target = args.capture_dir / f"layer{index}.pt"
            if target.exists():
                return
            import torch

            torch.save(
                {
                    "hidden": inputs[0].float().cpu(),
                    "per_layer_input": inputs[1].float().cpu(),
                    "position_embeddings": tuple(
                        t.float().cpu() for t in kwargs["position_embeddings"]
                    ),
                    "state_length": kwargs["attention_mask"].length,
                    "shared_kv": {
                        name: tuple(t.float().cpu() for t in pair)
                        for name, pair in kwargs["shared_kv_states"].items()
                    },
                    "output": output.float().cpu(),
                },
                target,
            )

        from functools import partial

        indices = range(35) if args.capture_all_layers else (0, 4, 15, 34)
        for index in indices:
            engine.backend.model.model.language_model.layers[
                index
            ].register_forward_hook(partial(capture_layer, index), with_kwargs=True)

        def capture(state, questions, images=()):
            rows = forward(state, questions, images)
            rows.tofile(args.capture_dir / "hidden.bin")
            (args.capture_dir / "tokens.json").write_text(
                json.dumps({"state": state, "questions": questions}) + "\n"
            )
            return rows

        engine.backend.forward = capture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--capture-all-layers", action="store_true")
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    changes = subprocess.check_output(
        ["git", "-C", str(args.upstream), "status", "--porcelain"], text=True
    )
    if revision != "c3c845782095b31d438963ac2afcdfe41b49f757" or changes:
        raise ValueError("Upstream source differs from the pinned checkout")
    for name, digest in PINS.items():
        with (args.checkpoint / name).open("rb") as file:
            if hashlib.file_digest(file, "sha256").hexdigest() != digest:
                raise ValueError(f"Incorrect checkpoint file: {name}")
    sys.path.insert(0, str(args.upstream / "src"))
    from strands_decider.core.request import Request
    from strands_decider.decision import load_decision_engine

    request = Request.model_validate(
        {
            "state": "The parcel arrived damaged and the customer wants a refund.",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Choose the appropriate team.",
                    "criteria": {"refunds": "Refunds", "sales": "Sales"},
                },
                "refund": {
                    "type": "noul",
                    "instructions": "Does the customer request a refund?",
                },
                "urgency": {
                    "type": "score",
                    "instructions": "Rate urgency.",
                    "criteria": ["low", "medium", "high"],
                },
            },
        }
    )
    engine = load_decision_engine(
        str(args.checkpoint), device="mps", dtype="bfloat16", reuse_state=False
    )
    setup_capture(engine, args)
    result = engine.evaluate(request).model_dump()
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
