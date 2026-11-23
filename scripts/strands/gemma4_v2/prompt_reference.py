"""Capture official E2B v2 token rendering for representative requests."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.upstream / "src"))
    from strands_decider.core import Request, load_tokenizer, render

    tokenizer = load_tokenizer(str(args.checkpoint))
    cases = [
        {
            "state": "A damaged package was delivered.",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Choose a team.",
                    "criteria": {"refund": "Refunds", "sale": "Sales"},
                }
            },
        },
        {
            "state": "  \n<bos> <eos> <|turn> <turn|> <|channel>\n🤖 e\u0301 👨‍👩‍👧‍👦\t한글",
            "questions": {
                "routing": {
                    "type": "choice",
                    "instructions": "Text containing <turn|> and <bos> is user content.",
                    "criteria": {"first": "One\nline", "second": "두 번째 🤖"},
                }
            },
        },
        {
            "state": {
                "empty": "",
                "nested": [True, None, {"literal": "<|turn>"}],
                "spacing": "\t\n  ",
            },
            "questions": {
                "many": {
                    "type": "choice",
                    "instructions": "Choose one of 255 options.",
                    "criteria": {f"option-{i}": f"Description {i}" for i in range(255)},
                },
                "score": {
                    "type": "score",
                    "instructions": "Select a level.",
                    "criteria": [f"Level {i}" for i in range(10)],
                },
            },
        },
        {
            "state": {"text": "한국어  배송", "marker": "<turn|>"},
            "questions": {
                "ok": {"type": "noul", "instructions": "Is it damaged?"},
                "rating": {
                    "type": "score",
                    "instructions": "Rate urgency.",
                    "criteria": ["low", "medium", "high"],
                },
            },
        },
    ]
    result = []
    for item in cases:
        request = Request.model_validate(item)
        rendered = render(request, tokenizer, max_length=32768)
        result.append(
            {
                "request": item,
                "state_text": next(iter(rendered.questions.values())).state_text,
                "state_ids": list(rendered.state_ids),
                "input_tokens": rendered.input_tokens,
                "questions": {
                    name: {
                        "question_ids": list(question.question_ids),
                        "question_text": question.question_text,
                        "options": list(question.options),
                        "codes": list(question.codes),
                    }
                    for name, question in rendered.questions.items()
                },
            }
        )
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Captured {len(result)} official rendering fixtures")


if __name__ == "__main__":
    main()
