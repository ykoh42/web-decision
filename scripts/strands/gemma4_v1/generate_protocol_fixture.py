"""Replay pinned upstream tokens and public answer rendering for the six BF16 CPU cases."""

import argparse
import hashlib
import json
from pathlib import Path

from strands_decider.prompting import (
    fit_window,
    option_token_index,
    render_question,
    render_state,
    to_answer,
)
from strands_decider.schema import SystemOneRequest
from tokenizers import Tokenizer

REFERENCE_SHA = "fa6646ff9df3d9664eb6add522b759d2c3ce38d6055d7dfa72bf46b23718f35d"
TOKENIZER_SHA = "30383f3ce90a169cbabe9eb3da7bb77988bf906aad2e59184b378f5470ec4dbd"

REQUESTS = {
    "duplicate-charge": {
        "state": "Customer was billed twice.",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"billing": None, "shipping": None},
            },
            "duplicate": {"type": "noul", "instructions": "Duplicate charge?"},
            "urgency": {
                "type": "score",
                "instructions": "Urgency?",
                "criteria": ["low", "medium", "high"],
            },
        },
    },
    "missing-package": {
        "state": {"package": "lost", "refund": False},
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"shipping": None, "billing": None},
            },
            "refunded": {"type": "noul", "instructions": "Refund sent?"},
            "risk": {
                "type": "score",
                "instructions": "Delivery risk?",
                "criteria": ["safe", "at risk", "failed"],
            },
        },
    },
}


def verify(path, sha):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError(f"Pinned file differs: {path.name}")
    return raw


def render_case(tok, key, original_request, answers):
    request = SystemOneRequest.model_validate(original_request)
    questions = [render_question(q) for q in request.questions.values()]
    state, ids, offsets = fit_window(
        tok, render_state(request.state), [q.text for q in questions], 4096
    )
    rows = {}
    token_pin = {"state": state, "questions": {}}
    for name, question, tokens, spans in zip(
        request.questions, questions, ids, offsets, strict=True
    ):
        index = option_token_index(spans, question.option_spans, 0)
        source = answers[key, name]
        probabilities = [source["probabilities"][slot] for slot in question.slot_labels]
        answer = to_answer(question, probabilities, ordinal_smoothing=0.1).model_dump()
        if answer != source["answer"]:
            raise ValueError("Upstream scoring revision differs on the pinned fixture")
        rows[name] = {
            "type": question.kind,
            "ids": tokens,
            "positions": index,
            "logits": source["logits"],
            "answer": answer,
        }
        token_pin["questions"][name] = [tokens, index]
    case = {
        "name": key,
        "request": original_request,
        "stateIds": state,
        "questions": rows,
        "usage": {
            "input_tokens": len(state) + sum(map(len, ids)),
            "output_tokens": len(questions),
        },
    }
    return case, token_pin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    tok = Tokenizer.from_str(verify(args.tokenizer, TOKENIZER_SHA).decode())
    original = [
        json.loads(row) for row in verify(args.reference, REFERENCE_SHA).splitlines()
    ]
    answers = {(row["case"], row["question"]): row for row in original[1:-1]}
    cases = []
    token_pin = {}
    for key, original_request in REQUESTS.items():
        case, token_pin[key] = render_case(tok, key, original_request, answers)
        cases.append(case)
    packed = json.dumps(token_pin, separators=(",", ":")).encode()
    if hashlib.sha256(packed).hexdigest() != original[0]["tokens_sha256"]:
        raise ValueError(
            "The two pinned upstream source revisions rendered different tokens"
        )
    with args.out.open("x") as file:
        json.dump({"referenceSha256": REFERENCE_SHA, "cases": cases}, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
