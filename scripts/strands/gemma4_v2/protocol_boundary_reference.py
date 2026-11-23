"""Capture protocol boundary and multimodal cases from the pinned upstream."""

import base64
import io
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path("../strands-upstream-v2/src").resolve()))
from strands_decider.core import Request, load_tokenizer, render


def image(width, height):
    out = io.BytesIO()
    Image.new("RGB", (width, height), (40, 90, 200)).save(out, format="PNG")
    return base64.b64encode(out.getvalue()).decode("ascii")


cases = []
for count in (2, 25, 26, 27, 206, 207, 208, 254, 255):
    cases.append(
        {
            "state": " \n  <|turn>한글🤖\t ",
            "questions": {
                "pick": {
                    "criteria": {
                        f"choice-{i}": None if i % 2 else f" 설명\n{i} "
                        for i in range(count)
                    },
                    "instructions": {"key": ["공백", None, 3]},
                }
            },
        }
    )
cases += [
    {
        "state": "",
        "questions": {
            "implicit": {"instructions": ""},
            "score": {"criteria": ["low", " high "], "instructions": " \n"},
        },
    },
    {
        "state": {
            "empty": [],
            "literal": "<|image|>",
            "nested": [True, False, None, 1e-7],
        },
        "questions": {
            "truth": {
                "type": "noul",
                "criteria": {"false": None},
                "instructions": "true?",
            }
        },
    },
    {
        "state": "multiple images",
        "images": [image(3, 3), image(10, 2)],
        "questions": {
            "vision": {
                "type": "choice",
                "instructions": "choose",
                "criteria": {"one": None, "two": " two "},
            }
        },
    },
]
tok = load_tokenizer("../strands-e2b-v2")
output = []
for item in cases:
    r = render(Request.model_validate(item), tok)
    output.append(
        {
            "request": item,
            "state_ids": list(r.state_ids),
            "images": [{"start": im.start, "end": im.end} for im in r.images],
            "input_tokens": r.input_tokens,
            "questions": {
                name: {
                    "question_ids": list(q.question_ids),
                    "question_text": q.question_text,
                    "options": list(q.options),
                    "codes": list(q.codes),
                }
                for name, q in r.questions.items()
            },
        }
    )
Path("../e2b-v2-protocol-boundary-reference.json").write_text(
    json.dumps(output, ensure_ascii=False), encoding="utf-8"
)
print("captured", len(output), "official protocol cases")
