"""Pinned official 1024-token text reference; --prepare-only uses CPU/tokenizer only."""

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

PINS = {
    "model.safetensors": "593a28a67754653c7ffffff80a5186f18966cc6262c2f4f097414c187e5fd48c",
    "readout.safetensors": "37b3f1c4670f1afb78007624e3ec2a2f55503d9b78602216e2f29a4cfb5111f6",
    "tokenizer.json": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
}
parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=Path)
parser.add_argument("upstream", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--prepare-only", action="store_true")
args = parser.parse_args()
revision = subprocess.check_output(
    ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
).strip()
dirty = subprocess.check_output(
    ["git", "-C", str(args.upstream), "status", "--porcelain"], text=True
)
if revision != "c3c845782095b31d438963ac2afcdfe41b49f757" or dirty:
    raise ValueError("Unpinned upstream source")
for name in ("tokenizer.json",) if args.prepare_only else PINS:
    with (args.checkpoint / name).open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != PINS[name]:
            raise ValueError(f"Unexpected file: {name}")
sys.path.insert(0, str(args.upstream / "src"))
render = importlib.import_module("strands_decider.core.prompt").render
Request = importlib.import_module("strands_decider.core.request").Request
load_tokenizer = importlib.import_module(
    "strands_decider.core.tokenizer"
).load_tokenizer

raw = {
    "state": " apple" * 955,
    "questions": {
        "q": {
            "type": "choice",
            "instructions": "Choose the relevant label.",
            "criteria": {"yes": "yes", "no": "no"},
        }
    },
}
request = Request.model_validate(raw)
rendered = render(request, load_tokenizer(str(args.checkpoint)))
question = rendered.questions["q"]
assert len(rendered.state_ids) + len(question.question_ids) == 1024
args.output.mkdir(parents=True, exist_ok=True)
(args.output / "request.json").write_text(json.dumps(raw) + "\n")
(args.output / "tokens.json").write_text(
    json.dumps({"state": rendered.state_ids, "questions": [question.question_ids]})
    + "\n"
)
print(
    json.dumps(
        {
            "state_tokens": len(rendered.state_ids),
            "question_tokens": len(question.question_ids),
            "total": 1024,
        }
    ),
    flush=True,
)
if not args.prepare_only:
    from strands_decider.decision import load_decision_engine

    engine = load_decision_engine(
        str(args.checkpoint), device="mps", dtype="bfloat16", reuse_state=False
    )
    result = engine.evaluate(request).model_dump()
    (args.output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)
