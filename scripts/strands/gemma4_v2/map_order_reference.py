"""Capture ordered dictionaries from the pinned upstream v2 prompt and readout."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path("../strands-upstream-v2/src").resolve()))
from strands_decider.core.prompt import render_text
from strands_decider.core.readout import answer
from strands_decider.core.request import Request

rows = []
for count in (2, 255):
    keys = [str(i) for i in reversed(range(count))]
    criteria = {key: f"description {key}" for key in keys}
    question = {"type": "choice", "instructions": "Choose", "criteria": criteria}
    state = {"10": {"20": "twenty", "2": "two"}, "2": ["value"]}
    req = Request.model_validate(
        {
            "state": state,
            "questions": {
                "10": question,
                "2": {"type": "noul", "instructions": "Is it true?"},
            },
        }
    )
    state_text, questions = render_text(req)
    probs = [0.0] * count
    probs[1] = 1.0
    q = SimpleNamespace(
        kind="choice",
        options=tuple(keys),
        request_options=tuple(keys),
        descriptions=tuple(criteria.values()),
    )
    rows.append(
        {
            "count": count,
            "keys": keys,
            "state_text": state_text,
            "questions": questions,
            "answer": answer(q, probs),
        }
    )
Path("packages/web-decision/test/fixtures/strands-gemma4-v2-map-order.json").write_text(
    json.dumps(rows, ensure_ascii=False) + "\n"
)
print("Captured upstream Map-order references", len(rows))
