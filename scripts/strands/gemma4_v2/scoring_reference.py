"""Capture official v2 probabilities and typed answers from deterministic logits."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path("../strands-upstream-v2/src").resolve()))
from strands_decider.core.readout import answer, code_probabilities, round_to_bf16

rng = np.random.default_rng(202611)


def generate_case(rng, kind, count, case):
    if case == 0:
        raw = np.zeros(count, dtype=np.float32)
    elif case == 1:
        raw = np.linspace(-90, 90, count, dtype=np.float32)
    elif case == 2:
        raw = np.ones(count, dtype=np.float32) * np.float32(1.234375)
    else:
        raw = rng.normal(0, 8, size=count).astype(np.float32)
    options = (
        ["false", "true"]
        if kind == "noul"
        else (
            [str(i) for i in range(count)]
            if kind == "score"
            else [f"option-{i}" for i in range(count)]
        )
    )
    descriptions = [f"level {i}" for i in range(count)]
    question = SimpleNamespace(
        kind=kind, options=options, request_options=options, descriptions=descriptions
    )
    probs = code_probabilities(round_to_bf16(raw), 30.0, 1.5456)
    return {
        "question": {"kind": kind, "options": options, "descriptions": descriptions},
        "logits": raw.tolist(),
        "probabilities": probs.tolist(),
        "answer": answer(question, probs.tolist()),
    }


def generate_rows(rng):
    rows = []
    for kind, count in [
        ("noul", 2),
        ("choice", 2),
        ("choice", 3),
        ("choice", 255),
        ("score", 2),
        ("score", 3),
        ("score", 10),
    ]:
        for case in range(12):
            rows.append(generate_case(rng, kind, count, case))
    return rows


rows = generate_rows(rng)
Path("packages/web-decision/test/fixtures/strands-gemma4-v2-scoring.json").write_text(
    json.dumps(rows, separators=(",", ":")) + "\n"
)
print("captured", len(rows), "official readout cases")
