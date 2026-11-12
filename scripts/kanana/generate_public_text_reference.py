#!/usr/bin/env python3
"""Generate pinned PyTorch Noul and Score answers for the public Kanana text request."""

import argparse
import hashlib
import json
from pathlib import Path

try:
    from pinned_model_files import verify_pinned_model_files
except ModuleNotFoundError:
    from scripts.kanana.pinned_model_files import verify_pinned_model_files

MODEL_ID = "kakaocorp/kanana-2-v-decide-3b-2611"
MODEL_REVISION = "12820d654ed9789a0f7f48daeee1703c3483c464"
CHECKPOINT_BYTES = 7_330_945_212
CHECKPOINT_SHA256 = "528930fa261b31ed1f47eecbe7d08efd2e226207508124ae186e1a65dbbbda3b"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = (
    Path.home()
    / ".cache/huggingface/hub/models--kakaocorp--kanana-2-v-decide-3b-2611"
    / "snapshots"
    / MODEL_REVISION
)
DEFAULT_BASE_FIXTURE = ROOT / "tests/fixtures/kanana-decision-python-reference.json"
DEFAULT_OUTPUT = ROOT / "tests/fixtures/kanana-public-text-python-reference.json"
PUBLIC_TEXT_QUESTIONS = {
    "needs_immediate_page": {
        "type": "noul",
        "instructions": "Did the parcel arrive late?",
        "criteria": {
            "false": "The parcel arrived on or before the promised date.",
            "true": "The parcel arrived after the promised date.",
        },
    },
    "delay_severity": {
        "type": "score",
        "instructions": "How severe was the delivery delay?",
        "criteria": ["No delay", "Small delay", "Significant delay"],
    },
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_fixture_pin(fixture):
    if (
        fixture.get("modelId") != MODEL_ID
        or fixture.get("revision") != MODEL_REVISION
        or fixture.get("checkpointSha256") != CHECKPOINT_SHA256
    ):
        raise SystemExit(
            "the public text fixture does not match the pinned Kanana checkpoint"
        )


def verify_choice_answer(actual, expected):
    if actual.get("choice") != expected.get("choice"):
        raise SystemExit(
            "the pinned PyTorch choice changed while generating public text answers"
        )
    for key, probability in expected["probabilities"].items():
        if abs(actual["probabilities"][key] - probability) >= 0.001:
            raise SystemExit(f"the pinned PyTorch choice probability changed for {key}")
    if abs(actual["confidence"] - expected["confidence"]) >= 0.001:
        raise SystemExit("the pinned PyTorch choice confidence changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--base-fixture", type=Path, default=DEFAULT_BASE_FIXTURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    import torch
    from transformers import AutoModel

    model_dir = args.model_dir.expanduser().resolve(strict=True)
    if model_dir.name != MODEL_REVISION:
        raise SystemExit(
            f"expected pinned model revision {MODEL_REVISION}: {model_dir}"
        )
    verify_pinned_model_files(model_dir)
    checkpoint = model_dir / "model.safetensors"
    if (
        checkpoint.stat().st_size != CHECKPOINT_BYTES
        or sha256_file(checkpoint) != CHECKPOINT_SHA256
    ):
        raise SystemExit(
            "the local checkpoint does not match the pinned Kanana artifact"
        )

    base_fixture = json.loads(args.base_fixture.read_text(encoding="utf-8"))
    verify_fixture_pin(base_fixture)
    request = base_fixture["request"]
    questions = {**request["questions"], **PUBLIC_TEXT_QUESTIONS}

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
        state=request["state"],
        questions=questions,
        state_format=request["state_format"],
        temperature=request["temperature"],
    )

    verify_choice_answer(
        result["answers"]["delivery_status"],
        base_fixture["result"]["answers"]["delivery_status"],
    )
    public_answers = {
        question_id: result["answers"][question_id]
        for question_id in PUBLIC_TEXT_QUESTIONS
    }
    public_fixture = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "checkpointSha256": CHECKPOINT_SHA256,
        "source": base_fixture["source"],
        "request": {
            "state": request["state"],
            "state_format": request["state_format"],
            "temperature": request["temperature"],
            "questions": PUBLIC_TEXT_QUESTIONS,
        },
        "result": {"answers": public_answers},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(public_fixture, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "fixture": str(args.output),
                "modelId": MODEL_ID,
                "revision": MODEL_REVISION,
                "checkpointSha256": CHECKPOINT_SHA256,
                "answers": public_answers,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
