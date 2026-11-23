"""Capture the official pinned E2B v2 model for a deterministic image request."""

import argparse
import base64
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PINS = {
    "model.safetensors": "593a28a67754653c7ffffff80a5186f18966cc6262c2f4f097414c187e5fd48c",
    "readout.safetensors": "37b3f1c4670f1afb78007624e3ec2a2f55503d9b78602216e2f29a4cfb5111f6",
    "tokenizer.json": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
}


def setup_capture(engine, args):
    model = engine.backend.model.model

    def capture_vision(_module, _inputs, kwargs, output):
        torch.save(
            {
                "pixel_values": kwargs["pixel_values"].float().cpu(),
                "pixel_position_ids": kwargs["pixel_position_ids"].cpu(),
                "output": output.last_hidden_state.float().cpu(),
            },
            args.output / "vision.pt",
        )

    def capture_embed(_module, _inputs, kwargs, output):
        torch.save(
            {
                "input": kwargs["inputs_embeds"].float().cpu(),
                "output": output.float().cpu(),
            },
            args.output / "embed-vision.pt",
        )

    model.vision_tower.register_forward_hook(capture_vision, with_kwargs=True)
    model.embed_vision.register_forward_hook(capture_embed, with_kwargs=True)

    def capture_layer(index, _module, inputs, kwargs, output):
        torch.save(
            {
                "hidden": inputs[0].float().cpu(),
                "position_embeddings": tuple(
                    value.float().cpu() for value in kwargs["position_embeddings"]
                ),
                "position_ids": kwargs["position_ids"].cpu(),
                "attention_mask": kwargs["attention_mask"].cpu()
                if kwargs["attention_mask"] is not None
                else None,
                "output": output.float().cpu(),
            },
            args.output / f"vision-layer{index}.pt",
        )

    from functools import partial

    for index in (0, 15):
        model.vision_tower.encoder.layers[index].register_forward_hook(
            partial(capture_layer, index), with_kwargs=True
        )

    def capture_text_layer(index, _module, inputs, kwargs, output):
        inputs[1].float().cpu().numpy().tofile(
            args.output / f"text-per-layer-{index}.bin"
        )
        if index in (0, 4, 34):
            torch.save(
                {
                    "hidden": inputs[0].float().cpu(),
                    "position_embeddings": tuple(
                        value.float().cpu() for value in kwargs["position_embeddings"]
                    ),
                    "output": output.float().cpu(),
                },
                args.output / f"text-layer{index}.pt",
            )

    for index in range(35):
        model.language_model.layers[index].register_forward_hook(
            partial(capture_text_layer, index), with_kwargs=True
        )
    original = engine.backend.forward

    def capture_forward(state, questions, images=()):
        (args.output / "tokens.json").write_text(
            json.dumps(
                {
                    "state": state,
                    "questions": questions,
                    "images": [{"start": im.start, "end": im.end} for im in images],
                }
            )
            + "\n"
        )
        rows = original(state, questions, images)
        rows.tofile(args.output / "hidden.bin")
        return rows

    engine.backend.forward = capture_forward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rev = subprocess.check_output(
        ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(args.upstream), "status", "--porcelain"], text=True
    )
    if rev != "c3c845782095b31d438963ac2afcdfe41b49f757" or dirty:
        raise ValueError("Unpinned upstream source")
    for name, expected in PINS.items():
        with (args.checkpoint / name).open("rb") as file:
            if hashlib.file_digest(file, "sha256").hexdigest() != expected:
                raise ValueError(f"Unexpected file: {name}")
    sys.path.insert(0, str(args.upstream / "src"))
    from strands_decider.core.request import Request
    from strands_decider.decision import load_decision_engine

    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    pixels[:, :32] = [220, 20, 30]
    pixels[:, 32:] = [15, 40, 210]
    image = Image.fromarray(pixels, mode="RGB")
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")
    request = Request.model_validate(
        {
            "state": "The image contains two color regions.",
            "images": [base64.b64encode(encoded.getvalue()).decode("ascii")],
            "questions": {
                "color": {
                    "type": "choice",
                    "instructions": "Which color is on the left?",
                    "criteria": {"red": "red", "blue": "blue"},
                }
            },
        }
    )
    args.output.mkdir(parents=True, exist_ok=True)
    engine = load_decision_engine(
        str(args.checkpoint), device="mps", dtype="bfloat16", reuse_state=False
    )
    if not engine.backend.takes_images:
        raise RuntimeError("Official reference image processor is unavailable")
    setup_capture(engine, args)
    result = engine.evaluate(request).model_dump()
    (args.output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
