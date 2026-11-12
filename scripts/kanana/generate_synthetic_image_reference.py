#!/usr/bin/env python3
"""Generate a pinned BF16 system_one reference for the Kanana synthetic image fixture."""

import argparse
import hashlib
import json
import math
from pathlib import Path

try:
    from pinned_model_files import verify_pinned_model_files
except ModuleNotFoundError:
    from scripts.kanana.pinned_model_files import verify_pinned_model_files

MODEL_ID = "kakaocorp/kanana-2-v-decide-3b-2611"
MODEL_REVISION = "12820d654ed9789a0f7f48daeee1703c3483c464"
CHECKPOINT_BYTES = 7_330_945_212
CHECKPOINT_SHA256 = "528930fa261b31ed1f47eecbe7d08efd2e226207508124ae186e1a65dbbbda3b"
IMAGE_BYTES = 4_735
IMAGE_SHA256 = "9a83fa578ba9d50c10d620fc49551648c581cff2c8fa7613fd7451798ad720cc"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = (
    Path.home()
    / ".cache/huggingface/hub/models--kakaocorp--kanana-2-v-decide-3b-2611"
    / "snapshots"
    / MODEL_REVISION
)
DEFAULT_IMAGE = ROOT / "tests/fixtures/kanana-synthetic-image.png"
DEFAULT_OUTPUT = ROOT / "tests/fixtures/kanana-synthetic-image-python-reference.json"
IMAGE_PATH = "tests/fixtures/kanana-synthetic-image.png"
STATE = "Inspect this synthetic illustration and decide which description fits best."
QUESTIONS = {
    "action": {
        "type": "choice",
        "instructions": "Which object is most prominent?",
        "criteria": {
            "rectangle": "A gold rectangle on the left is prominent.",
            "circle": "A teal circle on the right is prominent.",
        },
    }
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def verify_model_and_image(model_dir, image_path):
    if model_dir.name != MODEL_REVISION:
        raise SystemExit(
            f"expected pinned model revision {MODEL_REVISION}: {model_dir}"
        )
    verify_pinned_model_files(model_dir)

    checkpoint = model_dir / "model.safetensors"
    if checkpoint.stat().st_size != CHECKPOINT_BYTES:
        raise SystemExit(
            "the local checkpoint size does not match the pinned Kanana artifact"
        )
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise SystemExit(
            "the local checkpoint digest does not match the pinned Kanana artifact"
        )

    image_bytes = image_path.read_bytes()
    if (
        len(image_bytes) != IMAGE_BYTES
        or hashlib.sha256(image_bytes).hexdigest() != IMAGE_SHA256
    ):
        raise SystemExit("the synthetic image does not match its pinned PNG bytes")

    from PIL import Image

    with Image.open(image_path) as image:
        if image.format != "PNG" or image.size != (256, 256):
            raise SystemExit("the synthetic image must be a 256x256 PNG")


def run_pinned_model(model_dir, image_path):
    import torch
    from transformers import AutoModel

    torch.set_num_threads(1)
    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()
    model.to(device="cpu")
    result = model.system_one(
        state=STATE,
        questions=QUESTIONS,
        images=[str(image_path)],
        state_format="text",
        temperature=1.0,
    )
    return result


def validate_choice_answer(result):
    answer = result.get("answers", {}).get("action", {})
    probabilities = answer.get("probabilities")
    if answer.get("type") != "choice" or not isinstance(probabilities, dict):
        raise SystemExit(
            "the pinned PyTorch system_one call returned no choice probabilities"
        )
    if list(probabilities) != ["rectangle", "circle"]:
        raise SystemExit("the PyTorch image probabilities changed criteria order")
    if answer.get("choice") not in probabilities:
        raise SystemExit("the PyTorch image choice is not one of the pinned criteria")
    values = list(probabilities.values())
    if not all(isinstance(value, (int, float)) for value in values):
        raise SystemExit("the PyTorch image probabilities must be numeric")
    if not all(math.isfinite(value) for value in values):
        raise SystemExit("the PyTorch image probabilities must be finite")
    if abs(sum(values) - 1.0) > 1e-6:
        raise SystemExit("the PyTorch image probabilities must sum to one")
    confidence = answer.get("confidence")
    if (
        not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        raise SystemExit("the PyTorch image confidence must be a finite probability")
    return answer, probabilities, confidence


def create_reference(answer, probabilities, confidence):
    return {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "checkpointSha256": CHECKPOINT_SHA256,
        "source": {
            "runtime": "Python/PyTorch system_one",
            "parameterDtype": "bfloat16",
            "mode": "checkpoint_full",
        },
        "input": {
            "path": IMAGE_PATH,
            "bytes": IMAGE_BYTES,
            "sha256": IMAGE_SHA256,
            "width": 256,
            "height": 256,
        },
        "request": {
            "state": STATE,
            "state_format": "text",
            "temperature": 1.0,
            "questions": QUESTIONS,
        },
        "result": {
            "answer": answer["choice"],
            "probabilities": probabilities,
            "confidence": confidence,
        },
    }


def write_reference(output_path, reference):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(reference, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"fixture": str(output_path), **reference}, indent=2, ensure_ascii=False
        )
    )


def main():
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve(strict=True)
    image_path = args.image.expanduser().resolve(strict=True)
    verify_model_and_image(model_dir, image_path)
    result = run_pinned_model(model_dir, image_path)
    answer, probabilities, confidence = validate_choice_answer(result)
    reference = create_reference(answer, probabilities, confidence)
    write_reference(args.output, reference)


if __name__ == "__main__":
    main()
