#!/usr/bin/env python3
"""Save a pinned PyTorch decision fixture for full-model TVM parity checks."""

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
DEFAULT_MODEL_DIR = (
    Path.home()
    / ".cache/huggingface/hub/models--kakaocorp--kanana-2-v-decide-3b-2611"
    / "snapshots"
    / MODEL_REVISION
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_pinned_model(model_dir):
    import torch
    from transformers import AutoModel

    checkpoint = model_dir / "model.safetensors"
    if model_dir.name != MODEL_REVISION:
        raise SystemExit(
            f"expected pinned model revision {MODEL_REVISION}: {model_dir}"
        )
    verify_pinned_model_files(model_dir)
    if (
        checkpoint.stat().st_size != CHECKPOINT_BYTES
        or sha256_file(checkpoint) != CHECKPOINT_SHA256
    ):
        raise SystemExit(
            "the local checkpoint does not match the pinned Kanana artifact"
        )

    torch.set_num_threads(1)
    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()
    return model


def build_request():
    return {
        "state": {
            "incident": "Database connections are refusing every checkout request.",
            "impact": "Customers cannot complete purchases.",
        },
        "questions": {
            "action": {
                "type": "choice",
                "instructions": "What should the operator do now?",
                "criteria": {
                    "page_on_call": "Page the on-call engineer immediately.",
                    "wait_and_monitor": "Wait five minutes and monitor the service.",
                },
            },
            "severity": {
                "type": "score",
                "instructions": "How urgent is the incident?",
                "criteria": [
                    "Routine issue; continue normal monitoring.",
                    "Elevated issue; investigate during this shift.",
                    "Critical issue; page the on-call engineer now.",
                ],
            },
            "page_required": {
                "type": "noul",
                "instructions": "Does this incident require an immediate on-call page?",
                "criteria": {
                    "false": "No. The issue can be handled during normal monitoring.",
                    "true": "Yes. Page the on-call engineer immediately.",
                },
            },
        },
    }


def capture_embeddings(model, original_embed_messages, captured_batches, messages_list):
    vectors, tokens = original_embed_messages(messages_list)
    batch = []
    for messages, vector in zip(messages_list, vectors, strict=True):
        inputs = model._inputs(messages)
        if "pixel_values" in inputs:
            raise RuntimeError(
                "the text-only decision fixture unexpectedly contains image pixels"
            )
        batch.append(
            {
                "inputIds": inputs["input_ids"][0].tolist(),
                "referenceVector": vector.detach().float().cpu().tolist(),
            }
        )
    captured_batches.append(batch)
    return vectors, tokens


def candidate_keys(question):
    criteria = question["criteria"]
    if isinstance(criteria, dict):
        return list(criteria)
    return list(map(str, range(len(criteria))))


def capture_question_case(model, request, question_id, question):
    captured_batches = []
    original_embed_messages = model.embed_messages

    def capture(messages_list):
        return capture_embeddings(
            model, original_embed_messages, captured_batches, messages_list
        )

    model._candidate_cache.clear()
    model.embed_messages = capture
    try:
        result = model.system_one(request["state"], {question_id: question})
    finally:
        model.embed_messages = original_embed_messages

    keys = candidate_keys(question)
    if (
        len(captured_batches) != 2
        or len(captured_batches[0]) != 1
        or len(captured_batches[1]) != len(keys)
    ):
        raise RuntimeError(
            f"captured invalid query/candidate batches for {question_id}"
        )
    inputs = [captured_batches[0][0], *captured_batches[1]]
    if any(not item["inputIds"] for item in inputs):
        raise RuntimeError(
            f"the {question_id} decision fixture produced an empty token sequence"
        )
    return {
        "questionId": question_id,
        "question": question,
        "candidateKeys": keys,
        "answer": result["answers"][question_id],
        "inputs": [
            {"role": "query" if index == 0 else "candidate", **item}
            for index, item in enumerate(inputs)
        ],
    }


def create_cases(model, request):
    return [
        capture_question_case(model, request, question_id, question)
        for question_id, question in request["questions"].items()
    ]


def create_fixture(model, request, cases):
    return {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "checkpointSha256": CHECKPOINT_SHA256,
        "request": request,
        "decisionTemperature": float(model.config.decision_temperature),
        "embeddingDimension": int(model.config.embedding_dimension),
        "cases": cases,
    }


def write_fixture(output_path, fixture):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(fixture, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "inputLengths": [
                    [len(item["inputIds"]) for item in case["inputs"]]
                    for case in fixture["cases"]
                ],
                "answers": {
                    case["questionId"]: case["answer"] for case in fixture["cases"]
                },
            },
            indent=2,
        )
    )


def main():
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve(strict=True)
    model = load_pinned_model(model_dir)
    request = build_request()
    cases = create_cases(model, request)
    fixture = create_fixture(model, request, cases)
    write_fixture(args.output, fixture)


if __name__ == "__main__":
    main()
