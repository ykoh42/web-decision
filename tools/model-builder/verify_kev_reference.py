"""Reproduce the pinned Kev reference for one documented decision request."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

REPOSITORY = Path(__file__).resolve().parents[2]
FIXTURES = REPOSITORY / "tests" / "fixtures"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    if sha256(path) != expected:
        raise ValueError(f"Unexpected SHA-256 for {path.name}")


def token_ids(tokenizer: Tokenizer, request: dict) -> tuple[list[int], list[int]]:
    question = request["questions"][0]

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False).ids

    ids = [151659, *encode(request["state"]), 151660, *encode(question["instructions"])]
    indices = []
    for option in question["options"]:
        ids.extend([151648, *encode(option), 151649])
        indices.append(len(ids) - 1)
    ids.append(151661)
    return ids, indices


def probabilities(logits: list[float]) -> list[float]:
    values = [math.exp(logit - max(logits)) for logit in logits]
    total = sum(values)
    return [value / total for value in values]


def verify(model_dir: Path) -> dict:
    reference = json.loads(
        (FIXTURES / "kev06-upstream-onnx-reference.json").read_text()
    )
    baseline = json.loads((FIXTURES / "kev06-main-baseline.json").read_text())
    graph = model_dir / "onnx" / "model_q4f16.onnx"
    weights = model_dir / "onnx" / "model_q4f16.onnx_data"
    tokenizer_path = model_dir / "tokenizer.json"
    require_hash(graph, reference["graphSha256"])
    require_hash(weights, reference["weightsSha256"])
    require_hash(tokenizer_path, reference["tokenizerSha256"])
    ids, indices = token_ids(
        Tokenizer.from_file(str(tokenizer_path)), baseline["request"]
    )
    if ids != reference["tokenIds"] or indices != reference["optionIndices"]:
        raise ValueError("Reference token IDs or option positions differ")
    inputs = np.asarray([ids], dtype=np.int64)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(
        str(graph), sess_options=options, providers=["CPUExecutionProvider"]
    )
    result = session.run(
        None, {"input_ids": inputs, "attention_mask": np.ones_like(inputs)}
    )[0]
    logits = [float(result[0, index]) for index in indices]
    if any(
        abs(actual - expected) > 0.001
        for actual, expected in zip(logits, reference["logits"], strict=True)
    ):
        raise ValueError("Upstream CPU logits do not match the pinned reference")
    scores = probabilities(logits)
    if any(
        abs(actual - expected) > 0.000001
        for actual, expected in zip(scores, reference["probabilities"], strict=True)
    ):
        raise ValueError("Upstream CPU probabilities do not match the pinned reference")
    return {
        "scope": reference["scope"],
        "tokenIds": ids,
        "logits": logits,
        "probabilities": scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Directory containing the pinned ONNX model and tokenizer",
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.model_dir), indent=2))


if __name__ == "__main__":
    main()
