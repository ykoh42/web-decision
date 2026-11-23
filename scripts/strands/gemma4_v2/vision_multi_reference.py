"""Prepare a multi-image request and, on demand, capture pinned upstream output."""

import argparse
import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != "c3c845782095b31d438963ac2afcdfe41b49f757":
        raise ValueError("Unpinned Strands source")
    args.output.mkdir(parents=True, exist_ok=True)
    first = np.zeros((48, 96, 3), np.uint8)
    first[:, :48] = [220, 20, 30]
    first[:, 48:] = [15, 40, 210]
    second = np.zeros((96, 48, 3), np.uint8)
    second[:48, :] = [10, 180, 30]
    second[48:, :] = [240, 220, 15]
    images = []
    for index, pixels in enumerate((first, second)):
        path = args.output / f"image-{index}.png"
        Image.fromarray(pixels, mode="RGB").save(path)
        images.append(base64.b64encode(path.read_bytes()).decode())
    request = {
        "state": "There are two images, each with two colored regions.",
        "images": images,
        "questions": {
            "color": {
                "type": "choice",
                "instructions": "What color is at the top of the second image?",
                "criteria": {"green": "green", "yellow": "yellow"},
            }
        },
    }
    (args.output / "request.json").write_text(json.dumps(request) + "\n")
    print("Wrote the two-image fixture without using a GPU", flush=True)
    if not args.evaluate:
        return
    with (args.checkpoint / "model.safetensors").open("rb") as source:
        if (
            hashlib.file_digest(source, "sha256").hexdigest()
            != "593a28a67754653c7ffffff80a5186f18966cc6262c2f4f097414c187e5fd48c"
        ):
            raise ValueError("Checkpoint digest differs")
    sys.path.insert(0, str(args.upstream / "src"))
    from strands_decider.core.request import Request
    from strands_decider.decision import load_decision_engine

    engine = load_decision_engine(
        str(args.checkpoint), device="mps", dtype="bfloat16", reuse_state=False
    )
    original = engine.backend.forward

    def capture(state, questions, images=()):
        (args.output / "tokens.json").write_text(
            json.dumps(
                {
                    "state": state,
                    "questions": questions,
                    "images": [
                        {"start": image.start, "end": image.end} for image in images
                    ],
                }
            )
            + "\n"
        )
        return original(state, questions, images)

    engine.backend.forward = capture
    result = engine.evaluate(Request.model_validate(request)).model_dump()
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
